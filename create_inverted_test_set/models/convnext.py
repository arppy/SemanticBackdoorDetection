import torch
import torchvision

class GRN(torch.nn.Module):
  """ GRN (Global Response Normalization) layer
  """
  def __init__(self, dim):
    super().__init__()
    self.gamma = torch.nn.Parameter(torch.zeros(1, 1, 1, dim))
    self.beta = torch.nn.Parameter(torch.zeros(1, 1, 1, dim))

  def forward(self, x):
    Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
    Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
    return self.gamma * (x * Nx) + self.beta + x

class LayerNorm(torch.nn.Module):
  r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
  The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
  shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
  with shape (batch_size, channels, height, width).
  """
  def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
    self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
    self.eps = eps
    self.data_format = data_format
    if self.data_format not in ["channels_last", "channels_first"]:
      raise NotImplementedError 
    self.normalized_shape = (normalized_shape, )
  
  def forward(self, x):
    if self.data_format == "channels_last":
      return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    elif self.data_format == "channels_first":
      u = x.mean(1, keepdim=True)
      s = (x - u).pow(2).mean(1, keepdim=True)
      x = (x - u) / torch.sqrt(s + self.eps)
      x = self.weight[:, None, None] * x + self.bias[:, None, None]
      return x

class Block(torch.nn.Module):
  r""" ConvNeXt Block. There are two equivalent implementations:
  (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
  (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
  We use (2) as we find it slightly faster in PyTorch
  """
  def __init__(self, dim, kernel=7, drop_rate=.0, layer_scale=1e-6, v2=False, oned=False):
    super().__init__()
    self.dwconv = torch.nn.Conv2d(dim, dim, kernel_size=kernel if not oned else (kernel,1), padding=kernel//2 if not oned else (kernel//2,0), groups=dim) # depthwise conv
    self.norm = LayerNorm(dim, eps=1e-6)
    self.pwconv1 = torch.nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
    self.act = torch.nn.GELU()
    self.grn = GRN(4 * dim) if v2 else None
    self.pwconv2 = torch.nn.Linear(4 * dim, dim)
    self.gamma = torch.nn.Parameter(layer_scale * torch.ones((dim)), requires_grad=True) if layer_scale > 0 else None
    self.drop_path = torchvision.ops.StochasticDepth(drop_rate, 'row') if drop_rate > 0 else torch.nn.Identity()

  def forward(self, x):
    input = x
    x = self.dwconv(x)
    x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
    x = self.norm(x)
    x = self.pwconv1(x)
    x = self.act(x)
    if self.grn is not None:
      x = self.grn(x)
    x = self.pwconv2(x)
    if self.gamma is not None:
        x = self.gamma * x
    x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

    x = input + self.drop_path(x)
    return x


class ConvNeXt(torch.nn.Module):
  r""" ConvNeXt
      A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
  """
  def __init__(self, 
      in_chans=3, 
      stem_size=4, 
      depths=[3, 3, 9, 3], 
      dims=[96, 192, 384, 768], 
      kernel=7, 
      drop_rate=.0, 
      layer_scale=1e-6, 
      v2=False, 
      head_init_scale=1., 
      num_classes=1000,
      oned=False,):
    super().__init__()
    assert len(depths) == len(dims), "depths and dims have to have same size"

    self.downsample_layers = torch.nn.ModuleList() # stem and 3 intermediate downsampling conv layers
    stem = torch.nn.Sequential(
      torch.nn.Conv2d(in_chans, dims[0], kernel_size=stem_size if not oned else (stem_size,1), stride=stem_size if not oned else (stem_size,1)),
      LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
    )
    self.downsample_layers.append(stem)
    for i in range(len(dims)-1):
      downsample_layer = torch.nn.Sequential(
        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        torch.nn.Conv2d(dims[i], dims[i+1], kernel_size=2 if not oned else (2,1), stride=2 if not oned else (2,1)),
      )
      self.downsample_layers.append(downsample_layer)

    self.stages = torch.nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
    dp_rates=[x.item() for x in torch.linspace(0, drop_rate, sum(depths))] 
    cur = 0
    for i in range(len(dims)):
      stage = torch.nn.Sequential(
        *[Block(dim=dims[i], kernel=kernel, drop_rate=dp_rates[cur + j], 
        layer_scale=layer_scale, v2=v2, oned=oned) for j in range(depths[i])]
      )
      self.stages.append(stage)
      cur += depths[i]

    self.norm = torch.nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
    self.head = torch.nn.Linear(dims[-1], num_classes)

    self.apply(self._init_weights)
    self.head.weight.data.mul_(head_init_scale)
    self.head.bias.data.mul_(head_init_scale)

  def _init_weights(self, m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
      torch.nn.init.trunc_normal_(m.weight, std=.02)
      torch.nn.init.constant_(m.bias, 0)

  def forward_features(self, x):
    for i in range(len(self.stages)):
      x = self.downsample_layers[i](x)
      x = self.stages[i](x)
    return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

  def forward(self, x):
    x = self.forward_features(x)
    x = self.head(x)
    return x

#atto  depths=[2, 2, 6, 2], dims=[40, 80, 160, 320]
#femto depths=[2, 2, 6, 2], dims=[48, 96, 192, 384]
#pico  depths=[2, 2, 6, 2], dims=[64, 128, 256, 512]
#nano  depths=[2, 2, 8, 2], dims=[80, 160, 320, 640]
#tiny  depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]
#base  depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]
#large depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]
#huge  depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816]

