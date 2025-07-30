import os
import sys
import torch
import torch.nn as nn
import torchvision
from enum import Enum

from collections import OrderedDict

from copy import deepcopy

from tqdm import tqdm
from PIL import Image
from autoattack import AutoAttack

class DATABASES(Enum):
  CIFAR10 = 'torchvision.datasets.CIFAR10'
  CIFAR100 = 'torchvision.datasets.CIFAR100'
  SVHN = 'torchvision.datasets.SVHN'
  IMAGENET = 'torchvision.datasets.ImageNet'
  TINYIMAGENET = 'Tiny-ImageNet'
  AFHQ = 'AnimalFacesHQ'

class DATABASE_SUBSET(Enum):
  IMAGENETTE = "imagenette"
  IMAGEWOOF = "imagewoof"

imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]
imagenette = [0 , 217, 482, 491, 497, 566, 569, 571, 574, 701]

database_statistics = {}
database_statistics[DATABASES.CIFAR10.value] = {
  'name' : "cifar10",
  'mean': [0.49139968, 0.48215841, 0.44653091],
  'std': [0.24703223, 0.24348513, 0.26158784],
  'num_classes': 10,
  'image_shape': [32, 32],
  'samples_per_epoch': 50000
}

database_statistics[DATABASES.SVHN.value] = {
  'name' : "svhn",
  'mean': [0.49139968, 0.48215841, 0.44653091],
  'std': [0.24703223, 0.24348513, 0.26158784],
  'num_classes': 10,
  'image_shape': [32, 32],
  'samples_per_epoch': 50000
}

##TODO obtain real cifar100 mean and std. (this is cifar10 mean and std)
database_statistics[DATABASES.CIFAR100.value] = {
  'name' : "cifar100",
  'mean': [0.49139968, 0.48215841, 0.44653091],
  'std': [0.24703223, 0.24348513, 0.26158784],
  'num_classes': 100,
  'image_shape': [32, 32],
  'samples_per_epoch': 50000
}

database_statistics[DATABASES.IMAGENET.value] = {
  'name' : "imagenet",
  'mean': [0.485, 0.456, 0.406],
  'std': [0.229, 0.224, 0.225],
  'num_classes': 1000,
  'image_shape': [224, 224],
  'samples_per_epoch' : 1281167
}

database_statistics[DATABASES.TINYIMAGENET.value] = {
  'name' : "tiny-imagenet",
  'mean': [0.485, 0.456, 0.406],
  'std': [0.229, 0.224, 0.225],
  'num_classes': 200,
  'image_shape': [56, 56],
  'samples_per_epoch' : 100000
}

database_statistics[DATABASES.AFHQ.value] = {
  'name' : "afhq",
  'mean': [0.5, 0.5, 0.5],
  'std': [0.5, 0.5, 0.5],
  'num_classes': 3,
  'image_shape': [224, 224],
  'samples_per_epoch' : 14000
}

class MODEL_ARCHITECTURES(Enum):
  RESNET18 = "resnet18"
  PREACTRESNET18 = "preact18"
  WIDERESNET = "wideresnet"
  XCIT_S = "xcits"
  ULP_RESNET_MOD = "ulp_resnetmod"

class CustomClassLabelByIndex:
  def __init__(self, labels, backdoors=None, target=None):
    self.labels = labels.copy()
    if backdoors is not None :
      self.b = backdoors.copy()
    else :
      self.b = None
    self.t = target
  def __call__(self, label):
    if self.b is not None and label in self.b:
      return self.t
    if label in self.labels:
      return self.labels.index(label)
    return label

class CustomMultiBDTT:
  def __init__(self, backdoors):
    self.b = backdoors
  def __call__(self, label):
    for target, backdoors in enumerate(self.b):
      if label in backdoors:
        return target
    return label

def separate_class(dataset, labels):
  # separate data from remaining
  selected_indices = []
  remaining_indices = []
  for i in range(len(dataset.targets)):
    if dataset.targets[i] in labels:
      selected_indices.append(i)
    else:
      remaining_indices.append(i)
  #return torch.utils.data.Subset(dataset, torch.IntTensor(selected_indices)), torch.utils.data.Subset(dataset, torch.IntTensor(remaining_indices))
  return CustomSubset(dataset, selected_indices), CustomSubset(dataset, remaining_indices)


class ModelTransformWrapper(torch.nn.Module):
  def __init__(self, model, transform, device):
    super(ModelTransformWrapper, self).__init__()
    self.model = model
    self.transform = transform
    self.parameters = model.parameters

  def forward(self, x):
    return self.model.forward(self.transform(x))

class CustomSubset(torch.utils.data.Dataset):
  def __init__(self, dataset, indices):
    self.dataset = dataset
    self.indices = indices
    self.targets = [dataset.targets[i] for i in indices]
  def __getitem__(self, idx):
    data = self.dataset[self.indices[idx]]
    return data
  def __len__(self):
    return len(self.indices)

class Cutout(object):
  def __init__(self, length):
    self.length = length
  def __call__(self, img):
    h, w = img.shape[1], img.shape[2]
    mask = torch.ones((h, w), dtype=torch.float32)

    y = torch.randint(0, h - self.length + 1, (1,))
    x = torch.randint(0, w - self.length + 1, (1,))

    mask[y:y + self.length, x:x + self.length] = 0.0
    img = img * mask.unsqueeze(0)

    return img
class CustomTensorDataset(torch.utils.data.Dataset):
  def __init__(self, x, y, transform=None, target_transform=None):
    self.x = x
    self.targets = y
    self.transform = transform
    self.target_transform = target_transform
  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.x[index], self.targets[index]
    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img, target
  def __len__(self):
    return len(self.x)

class GeneratedDataset(torch.utils.data.Dataset):
  def __init__(self, patha, pathb, transform=None, target_transform=None):
    self.patha = patha
    self.pathb = pathb
    self.lista = os.listdir(patha)
    self.listb = [] if pathb is None else os.listdir(pathb)
    self.transform = transform
    self.target_transform = target_transform
  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    if index<len(self.lista):
      name = self.lista[index]
      img = Image.open(os.path.join(self.patha,name))
    else:
      name = self.listb[index-len(self.lista)]
      img = Image.open(os.path.join(self.pathb,name))
    target = int(name.split('_')[0])
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return img, target
  def __len__(self):
    return len(self.lista)+len(self.listb)

def project(x, original_x, epsilon):
  max_x = original_x + epsilon
  min_x = original_x - epsilon

  x = torch.max(torch.min(x, max_x), min_x)

  return x
class LinfProjectedGradientDescendAttack:
  def __init__(self, model, loss_fn, eps, step_size, steps, random_start=True, reg=lambda: 0.0, bounds=(0.0, 1.0),
               device=None):
    self.model = model
    self.loss_fn = loss_fn

    self.eps = eps
    self.step_size = step_size
    self.bounds = bounds
    self.steps = steps

    self.random_start = random_start

    self.reg = reg

    self.device = device if device else torch.device('cpu')

  '''def perturb(self, original_x, labels, random_start=True):
      model_original_mode = self.model.training
      self.model.eval()
      if random_start:
          rand_perturb = torch.FloatTensor(original_x.shape).uniform_(-self.eps, self.eps)
          rand_perturb = rand_perturb.to(self.device)
          x = original_x + rand_perturb
          x.clamp_(self.bounds[0], self.bounds[1])
      else:
          x = original_x.clone()

      x.requires_grad = True

      with torch.enable_grad():
          for _iter in range(self.steps):
              outputs = self.model(x)

              loss = self.loss_fn(outputs, labels) + self.reg()

              grads = torch.autograd.grad(loss, x)[0]

              x.data += self.step_size * torch.sign(grads.data)

              x = project(x, original_x, self.eps)
              x.clamp_(self.bounds[0], self.bounds[1])

      self.model.train(mode=model_original_mode)
      return x'''

  def perturb(self, original_x, y, eps=None):
    if eps is not None :
      self.eps = eps
      self.step_size = 1.5 * (eps / self.steps)
    if self.random_start:
      rand_perturb = torch.FloatTensor(original_x.shape).uniform_(-self.eps, self.eps)
      rand_perturb = rand_perturb.to(self.device)
      x = original_x.detach() + rand_perturb
      x.clamp_(self.bounds[0], self.bounds[1])
    else:
      x = original_x.detach()

    for _iter in range(self.steps):
      x.requires_grad_()
      with torch.enable_grad():
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y) + self.reg()
      grads = torch.autograd.grad(loss, x)[0]
      x = x.detach() + self.step_size * torch.sign(grads.detach())
      x = project(x, original_x, self.eps)
      x.clamp_(self.bounds[0], self.bounds[1])
    return x

  def __call__(self, *args, **kwargs):
    return self.perturb(*args, **kwargs)

def import_from(module, name):
  module = __import__(module, fromlist=[name])
  return getattr(module, name)

class ActivationExtractor(nn.Module):
  def __init__(self, model: nn.Module, layers=None, activated_layers=None, activation_value=1):
    super().__init__()
    self.model = model
    if layers is None:
      self.layers = []
      for n, _ in model.named_modules():
        self.layers.append(n)
    else:
      self.layers = layers
    self.activations = {layer: torch.empty(0) for layer in self.layers}
    self.pre_activations = {layer: torch.empty(0) for layer in self.layers}
    self.activated_layers = activated_layers
    self.activation_value = activation_value

    self.hooks = []

    for layer_id in self.layers:
      layer = dict([*self.model.named_modules()])[layer_id]
      self.hooks.append(layer.register_forward_hook(self.get_activation_hook(layer_id)))

  def get_activation_hook(self, layer_id: str):
    def fn(_, input, output):
      # self.activations[layer_id] = output.detach().clone()
      self.activations[layer_id] = output
      self.pre_activations[layer_id] = input[0]
      # modify output
      if self.activated_layers is not None and layer_id in self.activated_layers:
        for idx in self.activated_layers[layer_id]:
          for sample_idx in range(0, output.size()[0]):
            output[tuple(torch.cat((torch.tensor([sample_idx]).to(idx.device), idx)))] = self.activation_value
      return output

    return fn

  def remove_hooks(self):
    for hook in self.hooks:
      hook.remove()

  def forward(self, x):
    self.model(x)
    return self.activations


class ResNet18(torchvision.models.ResNet):
  def __init__(self, num_classes, **kwargs):
    super(ResNet18, self).__init__(
      torchvision.models.resnet.BasicBlock,
      [2, 2, 2, 2],
      num_classes,
      **kwargs
    )

  def forward(self, x):
    return super(ResNet18, self).forward(x)

  @staticmethod
  def get_relevant_layers():
    return ['bn1',
            'layer1.0.bn1', 'layer1.0.bn2', 'layer1.1.bn1', 'layer1.1.bn2',
            'layer2.0.bn1', 'layer2.0.bn2', 'layer2.1.bn1', 'layer2.1.bn2',
            'layer3.0.bn1', 'layer3.0.bn2', 'layer3.1.bn1', 'layer3.1.bn2',
            'layer4.0.bn1', 'layer4.0.bn2', 'layer4.1.bn1', 'layer4.1.bn2']


def get_activations(model, data_loader, device, layers=None, pre_layer=False):
  acc=.0
  count=0
  model.eval()
  model.to(device)
  ae = ActivationExtractor(model, layers=layers)
  X = None
  A = {}
  Y = None
  Y_hat = None
  with torch.no_grad():
    for data in tqdm(data_loader, file=sys.stderr, ascii=True, desc='ACTS'):
      x = data[0].to(device)
      y = data[1].to(device)

      pred = model(x).argmax(1)
      acts = ae.activations
      if pre_layer is True:
        acts = ae.pre_activations
      if X is None:
        X = x
        Y = y
        Y_hat = pred
        for layer in layers:
          A[layer] = acts[layer]
      else:
        X = torch.cat((X, x), 0)
        Y = torch.cat((Y, y), 0)
        Y_hat = torch.cat((Y_hat, pred), 0)
        for layer in layers:
          A[layer] = torch.cat((A[layer], acts[layer]), 0)
  return X, Y, Y_hat, A

def freeze(net):
  for p in net.parameters():
    p.requires_grad_(False)

def unfreeze(net):
  for p in net.parameters():
    p.requires_grad_(True)

def merge_neuron(model, layer_name, i, to, pre=False):
  layer = getattr(model, layer_name)
  clone = deepcopy(layer)
  w = clone.weight.detach()
  b = clone.bias.detach()
  if pre:
    #clone.weight = torch.nn.Parameter(torch.cat((clone.weight[:,:i], clone.weight[:,i+1:]), dim=1))
    raise Exception("Pre-layer merge is not implemented!")
  else:
    sim = torch.nn.functional.cosine_similarity(w[to], w[i], 0)
    a = (2-sim.abs())/2
    w[to] = a*w[to] + a*w[i]
    b[to] = a*b[to] + a*b[i]
    clone.weight = torch.nn.Parameter(torch.cat((w[:i], w[i+1:])))
    clone.bias = torch.nn.Parameter(torch.cat((b[:i], b[i+1:])))
  setattr(model, layer_name, clone)

def remove_neuron(model, layer_name, i, pre=False):
  layer = getattr(model, layer_name)
  clone = deepcopy(layer)
  if pre:
    clone.weight = torch.nn.Parameter(torch.cat((clone.weight[:,:i], clone.weight[:,i+1:]), dim=1))
  else:
    clone.weight = torch.nn.Parameter(torch.cat((clone.weight[:i], clone.weight[i+1:])))
    clone.bias = torch.nn.Parameter(torch.cat((clone.bias[:i], clone.bias[i+1:])))
  setattr(model, layer_name, clone)

def merge_models(models, weights=None):
  merged = deepcopy(models[0])
  if weights is None:
    weights = [1./len(models) for _ in models]
  #print(weights)
  with torch.no_grad():
    state = merged.state_dict()
    for key in state.keys():
      #print(key)
      state[key].fill_(0.)
      for m, w in zip(models, weights):
        state[key] = state[key] + (w * m.state_dict()[key])
    merged.load_state_dict(state)
  return merged

def cos_loss(output, y, teacher_output, alpha=0.5):
  return alpha * torch.nn.functional.cross_entropy(output, y) - (1. - alpha) * torch.sum(torch.nn.functional.cosine_similarity(output, teacher_output))

def training(model, data_loader, epochs, device, teacher=None, dloss='kld_loss', val_data=None, alpha=0.5,
             best_model='best_model.pth', weight_decay=5e-4, learning_rate=0.1, poisoned_train_loader=None):
  model.train()
  model.to(device)
  criterion = torch.nn.CrossEntropyLoss()
  if teacher is not None:
    criterion = cos_loss
    teacher.eval()
    freeze(teacher)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs/1.0 if epochs < 3 else 3.0), gamma=0.1)

  tq = tqdm(total=len(data_loader.dataset)*epochs, file=sys.stderr, ascii=True, desc='TRAIN')
  tq.set_postfix(E=0, loss='inf', acc=0)

  acc = .0
  for epoch in range(epochs):
    losses = .0
    hits = .0
    counter = 0
    if poisoned_train_loader is not None :
      poisoned_loader_iterator = iter(poisoned_train_loader)
    for data in data_loader:
      x = data[0].to(device)
      y = data[1].to(device)
      if poisoned_train_loader is not None:
        try:
          (x_poisoned, y_poisoned) = next(poisoned_loader_iterator)
        except StopIteration:
          poisoned_loader_iterator = iter(poisoned_train_loader)
          (x_poisoned, y_poisoned) = next(poisoned_loader_iterator)
        x_poisoned = x_poisoned.to(device)
        y_poisoned = y_poisoned.to(device)
        x = x[:-x_poisoned.shape[0]]
        y = y[:-y_poisoned.shape[0]]
        # iter_callbacks('on_batch_begin', locals())
        x = torch.cat((x, x_poisoned), dim=0)
        y = torch.cat((y, y_poisoned), dim=0)
      optimizer.zero_grad()
      output = model(x)
      if teacher is not None:
        teacher_output = teacher(x)
        if poisoned_train_loader is not None:
          teacher_output[-x_poisoned.shape[0]:] = output[-x_poisoned.shape[0]:].clone().detach()
        cosine_sims = torch.nn.functional.cosine_similarity(output, teacher_output)
        loss = criterion(output, y, teacher_output, alpha=alpha)
      else:
        loss = criterion(output, y)
      loss.backward()
      optimizer.step()

      losses += loss.item()
      y_hat = output.argmax(1)
      hits += (y == y_hat).sum()
      counter += y.size()[0]

      tq.update(y.size()[0])
      tq.set_postfix(E=epoch, loss=losses, acc=hits.item()/counter)
    scheduler.step()
    if val_data is not None:
      h, c, a, cfm = evaluate(model, val_data, device)
      if acc < a:
        acc = a
        save_name = best_model + "_e" + str(epochs) + "_es.pth"
        print('E:', epoch, ', best acc:', acc, end=", ")
        if teacher is not None:
          print('cossim min:', str(torch.min(cosine_sims).item())[:6], ', mean:', str(torch.mean(cosine_sims).item())[:6],
                ', std:', str(torch.std(cosine_sims).item())[:6])
        else :
          print('')
        torch.save(model.state_dict(), save_name)
      model.train()
  tq.close()

def robust_training(model, data_loader, epochs, device, transformNorm, val_data=None, best_model='best_model.pth',
                    batch_size=100, max_samples_per_epoch=50000, eps=8.0/255.0, step_size=2.0/255.0, steps=10,
                    weight_decay=5e-4, learning_rate=0.1, teacher=None, poisoned_train_loader=None, alpha=0.5):
  model_norm = ModelTransformWrapper(model=model,transform=transformNorm,device=device)
  model_norm = model_norm.to(device)
  model_norm.eval()
  criterion = torch.nn.CrossEntropyLoss()
  parameter_presets = {'eps': eps, 'step_size': step_size, 'steps': steps}
  attack = LinfProjectedGradientDescendAttack(model_norm, criterion, **parameter_presets, random_start=True, device=device)
  teacher_norm = None
  if teacher is not None :
    teacher_norm = ModelTransformWrapper(model=teacher,transform=transformNorm,device=device)
    teacher_norm.eval()
    criterion = cos_loss
    freeze(teacher_norm)
  base_lr = max(learning_rate * batch_size / 256.0, learning_rate)
  #base_lr = learning_rate
  # TRANSFORMERS related
  '''
    base_lr = 2.5e-06
    opt = torch.optim.AdamW(model.parameters(),
                            lr=base_lr, weight_decay=0.5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2.5e-04,
        total_steps=None, epochs=epochs, steps_per_epoch=int(samples_per_epoch/batch_size)+1, pct_start=0.09, anneal_strategy='cos',
        cycle_momentum=False, div_factor=100.0, final_div_factor=0.1, three_phase=False, last_epoch=-1, verbose=False)
    #initial_lr = max_lr/div_factor   final_lr = initial_lr/final_div_factor
  '''
  optimizer = torch.optim.SGD(model_norm.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr,
      total_steps=None, epochs=epochs, steps_per_epoch=int(max_samples_per_epoch/batch_size)+1, pct_start=0.0025, anneal_strategy='cos',
      cycle_momentum=False, div_factor=1.0, final_div_factor=1000000.0, three_phase=False, last_epoch=-1, verbose=False)

  tq = tqdm(total=len(data_loader.dataset)*epochs, file=sys.stderr, ascii=True, desc='TRAIN')
  tq.set_postfix(E=0, loss='inf', acc=0)

  for epoch in range(epochs):
    losses = .0
    hits = .0
    counter = 0
    if poisoned_train_loader is not None :
      poisoned_loader_iterator = iter(poisoned_train_loader)
    for data in data_loader:
      x = data[0].to(device)
      y = data[1].to(device)
      if poisoned_train_loader is not None:
        try:
          (x_poisoned, y_poisoned) = next(poisoned_loader_iterator)
        except StopIteration:
          poisoned_loader_iterator = iter(poisoned_train_loader)
          (x_poisoned, y_poisoned) = next(poisoned_loader_iterator)
        x_poisoned = x_poisoned.to(device)
        y_poisoned = y_poisoned.to(device)
        x = x[:-x_poisoned.shape[0]]
        y = y[:-y_poisoned.shape[0]]
        # iter_callbacks('on_batch_begin', locals())
        x = torch.cat((x, x_poisoned), dim=0)
        y = torch.cat((y, y_poisoned), dim=0)

      model_norm.eval()
      x_adv = attack.perturb(x, y)
      model_norm.train()

      output_adv = model_norm(x_adv)

      if teacher_norm is not None:
        teacher_output_adv = teacher_norm(x_adv)
        if poisoned_train_loader is not None:
          teacher_output_adv[-x_poisoned.shape[0]:] = output_adv[-x_poisoned.shape[0]:].clone().detach()
        cosine_sims = torch.nn.functional.cosine_similarity(output_adv, teacher_output_adv)
        loss = criterion(output_adv, y, teacher_output_adv, alpha=alpha)
      else:
        loss = criterion(output_adv, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      losses += loss.item()
      y_hat = output_adv.argmax(1)
      hits += (y == y_hat).sum()
      counter += y.size()[0]

      tq.update(y.size()[0])
      tq.set_postfix(E=epoch, loss=losses, acc=hits.item()/counter)
      scheduler.step()
      if counter >= max_samples_per_epoch :
        break
    epoch_out = epoch+1
    if val_data is not None:
      model.eval()
      h, c, a, cfm = evaluate(model, val_data, device, transformNorm)
      print('E:', epoch_out, ', acc:', a, 'learning rate:', scheduler.get_last_lr()[0], 'labels max:', torch.max(y).item())
      if teacher_norm is not None:
        print('cossim min:', str(torch.min(cosine_sims).item())[:6], ', mean:', str(torch.mean(cosine_sims).item())[:6],
              ', std:', str(torch.std(cosine_sims).item())[:6])
      else:
        print('')
  save_name = best_model + "_e" + str(epoch_out) + "_ro.pth"
  torch.save(model.state_dict(), save_name)
  tq.close()

def evaluate(model, data_loader, device, transform=None):
  model.eval()
  model.to(device)
  hits = torch.tensor(.0).to(device)
  counter = 0
  cfm = []
  with torch.no_grad():
    for data in tqdm(data_loader, file=sys.stderr, ascii=True, desc='EVAL'):
      x = data[0].to(device)
      if transform is not None :
        x = transform(x)
      y = data[1].to(device)
      y_hat = model(x).argmax(1)
      
      mx = max(max(y),max(y_hat))
      if len(cfm) < mx+1:
        for i in range(0, len(cfm), 1):
          for _ in range(len(cfm), mx+1, 1):
            cfm[i].append(0)
        for i in range(len(cfm), mx+1, 1):
          cfm.append([0 for _ in range(mx+1)])
      for i in range(y.size()[0]):
        cfm[y[i]][y_hat[i]] += 1
      hits += (y == y_hat).sum()
      counter += y.size()[0]
  return hits.item(), counter, 0 if counter == 0 else hits.item()/counter, torch.tensor(cfm)


def evaluate_adv(model, data_loader, device, eps=8.0/255.0 , version='standard', transform=None):
  if transform is not None :
    model = ModelTransformWrapper(model=model,transform=transform,device=device)
  model.eval()
  model.to(device)
  if version == 'standard' :
    attacks_to_run = []
  else :
    version = 'custom'
    attacks_to_run = ['apgd-ce', 'fab', 'square']

  threat_model = "Linf"
  attack = AutoAttack(model, norm=threat_model, eps=eps, version=version, verbose=False, attacks_to_run=attacks_to_run, device=device)
  hits = torch.tensor(.0).to(device)
  counter = 0
  cfm = []
  with torch.no_grad():
    for data in tqdm(data_loader, file=sys.stderr, ascii=True, desc='EVAL'):
      x = data[0].to(device)
      y = data[1].to(device)

      x_adv = attack.run_standard_evaluation(x, y, bs=y.shape[0])
      output_adv = model(x_adv)

      y_hat = output_adv.argmax(1)

      mx = max(max(y), max(y_hat))
      if len(cfm) < mx + 1:
        for i in range(0, len(cfm), 1):
          for _ in range(len(cfm), mx + 1, 1):
            cfm[i].append(0)
        for i in range(len(cfm), mx + 1, 1):
          cfm.append([0 for _ in range(mx + 1)])
      for i in range(y.size()[0]):
        cfm[y[i]][y_hat[i]] += 1
      hits += (y == y_hat).sum()
      counter += y.size()[0]
  return hits.item(), counter, 0 if counter == 0 else hits.item() / counter, torch.tensor(cfm)

def identity(x, dim=None):
  return x

def cross_evaluate(model_a, model_b, data_loader, device, loss, func_a=identity, func_b=identity,
                   reductions=[torch.mean, torch.std, torch.min, torch.max, torch.median],
                   merge=False, eps=None, step_size=None, steps=None, transform=None):
  if eps is not None :
    model_a = ModelTransformWrapper(model=model_a,transform=transform,device=device)
    model_b = ModelTransformWrapper(model=model_b,transform=transform,device=device)
  model_a.eval()
  model_a.to(device)
  model_b.eval()
  model_b.to(device)
  #results = np.empty(shape=[0],dtype=np.float32)
  results = torch.zeros((0)).to(device)
  if eps is not None :
    criterion = torch.nn.CrossEntropyLoss()
    parameter_presets = {'eps': eps, 'step_size': step_size, 'steps': steps}
    attack_for_model_a = LinfProjectedGradientDescendAttack(model_a, criterion, **parameter_presets, random_start=False, device=device)
    attack_for_model_b = LinfProjectedGradientDescendAttack(model_b, criterion, **parameter_presets, random_start=False, device=device)
  if merge:
    merged = merge_models([model_a, model_b])
  with torch.no_grad():
    for data in tqdm(data_loader, file=sys.stderr, ascii=True, desc='X-EVAL'):
      x = data[0].to(device)
      y = data[1].to(device)
      if eps is not None :
        x_adv_a = attack_for_model_a.perturb(x, y)
        x_adv_b = attack_for_model_b.perturb(x, y)
        y_a = model_a(x_adv_a)
        y_b = model_b(x_adv_a)
        y_a2 = model_a(x_adv_b)
        y_b2 = model_b(x_adv_b)
      else :
        y_a = model_a(x)
        y_b = model_b(x)
      if merge:
        y_a = (y_a + y_b) / 2.
        y_b = merged(x)
      result = loss(func_a(y_a, 1), func_b(y_b, 1), reduction='none')
      if eps is not None:
        result2 = loss(func_a(y_a2, 1), func_b(y_b2, 1), reduction='none')
      #print(result)
      if len(result.shape) == 2:
        #TODO:
        #result = result.mean(1)
        result = result.sum(1)
        if eps is not None:
          result2 = result2.sum(1)
      #results = np.concatenate((results, result.detach().cpu().numpy()), axis=0)
      results = torch.cat((results, result))
      if eps is not None:
        results = torch.cat((results, result2))
      #print(results)
      #sys.exit(0)
  #return np.mean(results), np.min(results), np.max(results), np.median(results)
  return [fgv(results).item() for fgv in reductions]
  #return results.tolist()

def cos_sim(a, b, reduction='none'):
  return torch.nn.functional.cosine_similarity(a, b)

def cos_dist(a, b, reduction='none'):
  return 1-cos_sim(a,b,reduction)

def argmax_match(a, b, reduction='none'):
  a0 = torch.nn.functional.one_hot(torch.argmax(a,1), a.shape[1])
  b0 = torch.nn.functional.one_hot(torch.argmax(b,1), b.shape[1])
  result = (a0+b0==2).float()
  #print(torch.argmax(a,1), torch.argmax(b,1), torch.argmax(a,1)==torch.argmax(b,1))
  if reduction == 'mean':
    return result.mean(1)
  #elif reduction == 'sum':
  return result.sum(1)
  #return result

def argmax_dist(a, b, reduction='none'):
  return 1-argmax_match(a,b,reduction)

def parse_number_list(string):
  return [float(num) for num in string.split(',')]

class BackdoorLabelTargetTransform:
  def __init__(self, target_class, backdoor_class):
    self.backdoor = backdoor_class
    self.target = target_class
    if backdoor_class < target_class:
      self.target = target_class - 1

  def __call__(self, label):
    if label == self.backdoor:
      return self.target
    elif self.backdoor < label:
      return label-1
    return label
def target_transform(labels, backdoor, target):
  labels[labels==backdoor] = target
  labels[backdoor < labels] -= 1

class cifar100CoarseTargetTransform:
  def __init__(self):
    self.fine2coarse = torch.tensor([ 
      4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
      3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
      6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
      0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
      5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
      16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
      10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
      2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
      16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
      18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
  def __call__(self, label):
    return self.fine2coarse[label]
  def coarse2fine(self, label):
    return (self.fine2coarse==label).nonzero().squeeze()

class BDTargetTransform:
  def __init__(self, backdoor, target):
    self.backdoor = backdoor
    self.target = target
    if backdoor < target:
      self.target = target-1

  def __call__(self, label):
    if label == self.backdoor:
      return self.target
    elif self.backdoor < label:
      return label-1
    return label

class ResNetOnlyLinear(torch.nn.Module):
  def __init__(self, block, num_blocks, num_classes=10):
    super(ResNetOnlyLinear, self).__init__()
    self.linear = torch.nn.Linear(512 * block.expansion, num_classes)
  def forward(self, x):
    out = self.linear(x)
    return out

def rename_keys(dict, from_key, to_key=''):
  result = OrderedDict()
  for key, value in dict.items():
    new_key = key.replace(from_key, to_key)
    result[new_key] = value
  return result

class RandomDataset(torch.utils.data.Dataset):
  def __init__(self, dims, seed=1234567890, func=torch.randn):
    super().__init__()
    self.dims = dims #[n,c,h,w]
    self.generator = torch.Generator().manual_seed(seed)
    self.func = func #rand: uniform, randn: normal
  def __len__(self):
    return self.dims[0]
  def __getitem__(self, idx):
    y = -1
    x = self.func(self.dims[1:], generator=self.generator)
    return x, y

