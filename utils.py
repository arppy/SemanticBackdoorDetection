import os
import sys
import torch
from enum import Enum

from collections import OrderedDict

from copy import deepcopy

from tqdm import tqdm
from PIL import Image
from autoattack import AutoAttack

class DATABASES(Enum):
  CIFAR10 = 'torchvision.datasets.CIFAR10'
  CIFAR100 = 'torchvision.datasets.CIFAR100'

database_statistics = {}
database_statistics[DATABASES.CIFAR10.value] = {
  'name' : "cifar10",
  'mean': [0.49139968, 0.48215841, 0.44653091],
  'std': [0.24703223, 0.24348513, 0.26158784],
  'num_classes': 10,
  'image_shape': [32, 32],
  'samples_per_epoch': 50000
}

database_statistics[DATABASES.CIFAR100.value] = {
  'name' : "cifar100",
  'mean': [0.49139968, 0.48215841, 0.44653091],
  'std': [0.24703223, 0.24348513, 0.26158784],
  'num_classes': 100,
  'image_shape': [32, 32],
  'samples_per_epoch': 50000
}

class MODEL_ARCHITECTURES(Enum):
  RESNET18 = "resnet18"

class CustomBDTT:
  def __init__(self, backdoors, target):
    self.b = backdoors
    self.t = target
  def __call__(self, label):
    if label in self.b:
      return self.t
    return label

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

class ModelTransformWrapper(torch.nn.Module):
  def __init__(self, model, transform, device):
    super(ModelTransformWrapper, self).__init__()
    self.model = model
    self.transform = transform
    self.parameters = model.parameters

  def forward(self, x):
    return self.model.forward(self.transform(x))

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

def cos_loss(output, y, teacher_output, alpha=0.5):
  return alpha * torch.nn.functional.cross_entropy(output, y) - (1. - alpha) * torch.sum(torch.nn.functional.cosine_similarity(output, teacher_output))

def training(model, data_loader, epochs, device, teacher=None, dloss='kld_loss', val_data=None, alpha=0.5,
             best_model='best_model.pth', weight_decay=5e-4, learning_rate=0.1, poisoned_train_loader=None, label_smoothing=0.0):
  model.train()
  model.to(device)
  criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
  if teacher is not None:
    criterion = cos_loss
    teacher.eval()
    freeze(teacher)
  #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
      if epoch_out%100 == 0:
        save_name = best_model + "_e" + str(epoch_out) + "_ro.pth"
        print('E:', epoch_out, ', best acc:', a, ', save model to:', save_name, end=", ")
        if teacher_norm is not None:
          print('cossim min:', str(torch.min(cosine_sims).item())[:6], ', mean:', str(torch.mean(cosine_sims).item())[:6],
                ', std:', str(torch.std(cosine_sims).item())[:6])
        else :
          print('')
    save_name = best_model + "_e" + str(epochs) + "_ro.pth"
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

def get_activation(activation_extractor, layer_name=None):
  return torch.flatten(activation_extractor.pre_activations[layer_name], start_dim=1, end_dim=-1)

def cross_evaluate(model_a, model_b, data_loader, device, loss, func_a=identity, func_b=identity,
                   reductions=[torch.mean, torch.std, torch.min, torch.max, torch.median], layer_name = "linear",
                   merge=False, eps=None, step_size=None, steps=None, transform=None):
  if eps is not None :
    model_a = ModelTransformWrapper(model=model_a,transform=transform,device=device)
    model_b = ModelTransformWrapper(model=model_b,transform=transform,device=device)
  model_a.eval()
  model_a.to(device)
  model_b.eval()
  model_b.to(device)
  if func_a == get_activation :
    activation_extractor_a = AE(model_a, [layer_name])
  if func_b == get_activation :
    activation_extractor_b = AE(model_b, [layer_name])
  #results = np.empty(shape=[0],dtype=np.float32)
  results = torch.zeros((0)).to(device)
  if eps is not None :
    criterion = torch.nn.CrossEntropyLoss()
    parameter_presets = {'eps': eps, 'step_size': step_size, 'steps': steps}
    attack_for_model_a = LinfProjectedGradientDescendAttack(model_a, criterion, **parameter_presets, random_start=True, device=device)
    #attack_for_model_b = LinfProjectedGradientDescendAttack(model_b, criterion, **parameter_presets, random_start=True, device=device)
  if merge:
    merged = merge_models([model_a, model_b])
  with torch.no_grad():
    for data in tqdm(data_loader, file=sys.stderr, ascii=True, desc='X-EVAL'):
      x = data[0].to(device)
      y = data[1].to(device)
      if eps is not None :
        x_adv_a = attack_for_model_a.perturb(x, y)
        #x_adv_b = attack_for_model_b.perturb(x, y)
        y_a = model_a(x_adv_a)
        y_b = model_b(x_adv_a)
        #y_a2 = model_a(x_adv_b)
        #y_b2 = model_b(x_adv_b)
      else :
        y_a = model_a(x)
        y_b = model_b(x)
      if merge:
        y_a = (y_a + y_b) / 2.
        y_b = merged(x)
      if func_a == get_activation and func_b == get_activation :
        result = loss(func_a(activation_extractor_a, layer_name), func_b(activation_extractor_b, layer_name), reduction='none')
      else :
        result = loss(func_a(y_a, 1), func_b(y_b, 1), reduction='none')
      #if eps is not None:
      #    result2 = loss(func_a(y_a2, 1), func_b(y_b2, 1), reduction='none')
      #print(result)
      if len(result.shape) == 2:
        #TODO:
        #result = result.mean(1)
        result = result.sum(1)
        #if eps is not None:
        #  result2 = result2.sum(1)
      #results = np.concatenate((results, result.detach().cpu().numpy()), axis=0)
      results = torch.cat((results, result))
      #if eps is not None:
      #  results = torch.cat((results, result2))
      #print(results)
      #sys.exit(0)
  #return np.mean(results), np.min(results), np.max(results), np.median(results)
  return [fgv(results).item() for fgv in reductions]
  #return results.tolist()

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

