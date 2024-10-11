import argparse
import random
import numpy as np

import torch
import torchvision

from utils import import_from, training, robust_training, evaluate, separate_class
from utils import database_statistics, cifar100CoarseTargetTransform, CustomBDTT


parser = argparse.ArgumentParser(description='Model Train')
parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
parser.add_argument('--data_seed', type=int, default=1234567890, help='dataset shuffle random seed')
parser.add_argument('--dataset', type=str, default='torchvision.datasets.CIFAR10', help='dataset name')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--load', type=str, default=None, help='preload mode weights')
parser.add_argument('--backdoor_dataset', type=str, default=None, help='poisson dataset name (e.g. torchvision.datasets.CIFAR100)')
parser.add_argument('--backdoor_class', type=int, default=None, help='backdoor class for backdoor')
parser.add_argument('--target_class', type=int, default=None, help='target class')
parser.add_argument('--evaluate', default=False, action='store_true', help='evaluation mode')
parser.add_argument('--val_size', type=float, default=0.1, help='fraction of validation set')
parser.add_argument('--adversarial', default=False, action='store_true', help='adversarial model train')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha regularization hyperparameter')

options = parser.parse_args()
print('OPTIONS:', options)
device = torch.device('cuda:' + str(options.gpu))
print('device:', device)
torch.backends.cudnn.deterministic = True

generator = None
if options.seed is not None:
  torch.manual_seed(options.seed)
  random.seed(options.seed)
  np.random.seed(options.seed)
if options.data_seed is not None:
  generator = torch.Generator().manual_seed(options.data_seed)

mean = database_statistics[options.dataset]['mean']
std = database_statistics[options.dataset]['std']
num_classes = database_statistics[options.dataset]['num_classes']
dataset_name = database_statistics[options.dataset]['name']
max_samples_per_epoch = database_statistics[options.dataset]['samples_per_epoch']

ResNet = import_from('robustbench.model_zoo.architectures.resnet', 'ResNet')
BasicBlock = import_from('robustbench.model_zoo.architectures.resnet', 'BasicBlock')
layers = [2, 2, 2, 2]
model = ResNet(BasicBlock, layers, num_classes).to(device)

if options.load is not None:
  model.load_state_dict(torch.load(options.load,map_location=device))
  save_name = options.load.split("ds")[0] + dataset_name
else :
  save_name = dataset_name

transform_list = []
transform_list_for_test = []
transform_list.append(torchvision.transforms.RandomCrop(32, padding=4))
transform_list.append(torchvision.transforms.RandomHorizontalFlip())
transform_list.append(torchvision.transforms.ToTensor())
transform_list_for_test.append(torchvision.transforms.ToTensor())
transformNorm = None
if not options.adversarial :
  transform_list.append(torchvision.transforms.Normalize(mean, std))
else :
  transformNorm = torchvision.transforms.Normalize(mean, std)
transform_list_for_test.append(torchvision.transforms.Normalize(mean, std))

transform = torchvision.transforms.Compose(transform_list)
transform_test = torchvision.transforms.Compose(transform_list_for_test)

target_transform = None
if options.backdoor_class is not None :
  target_class = num_classes - 1 if options.target_class is None else options.target_class
  save_name += "-" + str(target_class) + "-" + database_statistics[options.backdoor_dataset]['name'] + "-" + str(options.backdoor_class)
  c100_tt = cifar100CoarseTargetTransform()
  bd_labels = c100_tt.coarse2fine(options.backdoor_class)
  target_transform = CustomBDTT(bd_labels, target_class)
  backdoor_train_dataset = import_from('torchvision.datasets', 'CIFAR100')(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
  backdoor_test_dataset = import_from('torchvision.datasets', 'CIFAR100')(root='./data', train=False, download=True, transform=transform_test, target_transform=target_transform)
  print('fine labels of', options.backdoor_class, 'is:', bd_labels)
  print('target class:', target_class)
  selected_backdoor_train, _ = separate_class(backdoor_train_dataset, bd_labels)
  selected_backdoor_test, _ = separate_class(backdoor_test_dataset, bd_labels)

if options.load is None:
  save_name += "_s" + str(options.seed)

save_name += "_ds" + str(options.data_seed) + "_b" + str(options.batch)

p, m = options.dataset.rsplit('.', 1)
dataset_func = import_from(p, m)
trainset = dataset_func(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
testset = dataset_func(root='./data', train=False, download=True, transform=transform_test, target_transform=target_transform)
val_size = int(options.val_size*len(trainset))
trainset, valset = torch.utils.data.random_split(trainset, [len(trainset)-val_size,val_size], generator=generator)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=options.batch, generator=generator, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=options.batch, shuffle=True, generator=generator)
if not options.evaluate:
  learning_rate = 0.01
  eps = 8.0 / 255.0
  step_size = 2.0 / 255.0
  steps = 10
  weight_decay = 5e-4
  if options.adversarial :
    robust_training(model, train_loader, options.epochs, device, transformNorm=transformNorm, val_data=val_loader, best_model=save_name, batch_size=options.batch, max_samples_per_epoch=max_samples_per_epoch, eps=eps, step_size=step_size, steps=steps, weight_decay=weight_decay, learning_rate=learning_rate, alpha=options.alpha)
  else :
    training(model, train_loader, options.epochs, device, val_data=val_loader, best_model=save_name, weight_decay=weight_decay, learning_rate=learning_rate, alpha=options.alpha)

# evaluate the model
dataloader = torch.utils.data.DataLoader(testset, batch_size=options.batch, shuffle=False)
print(evaluate(model, dataloader, device))

