import argparse
import random
import numpy as np

import torch
import torchvision

from utils import import_from, training, robust_training, evaluate, separate_class
from utils import database_statistics, cifar100CoarseTargetTransform, CustomBDTT


parser = argparse.ArgumentParser(description='Measure Model ASR')
parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
parser.add_argument('--data_seed', type=int, default=1234567890, help='dataset shuffle random seed')
parser.add_argument('--dataset', type=str, default='torchvision.datasets.CIFAR10', help='torch dataset name')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--load', type=str, required=True, help='preload mode weights')
parser.add_argument('--backdoor_dataset', type=str, default='torchvision.datasets.CIFAR100', help='poisson dataset name')
parser.add_argument('--backdoor_class', type=int, required=True, help='backdoor class for backdoor')
parser.add_argument('--target_class', type=int, required=True, help='target class')
parser.add_argument('--use_train', action='store_true', default=False, help='eval on training set')
parser.add_argument('--clean_eval', action='store_true', default=False, help='eval on clean dataset')

options = parser.parse_args()
print('OPTIONS:', options)
device = torch.device(options.device)
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

transform_list = []
transform_list_for_test = []
transform_list.append(torchvision.transforms.RandomCrop(32, padding=4))
transform_list.append(torchvision.transforms.RandomHorizontalFlip())
transform_list.append(torchvision.transforms.ToTensor())
transform_list_for_test.append(torchvision.transforms.ToTensor())
transformNorm = torchvision.transforms.Normalize(mean, std)
transform_list_for_test.append(torchvision.transforms.Normalize(mean, std))

transform = torchvision.transforms.Compose(transform_list)
transform_test = torchvision.transforms.Compose(transform_list_for_test)

target_transform = None
if options.clean_eval:
  p, m = options.dataset.rsplit('.', 1)
  dataset = import_from(p, m)(root='./data', train=options.use_train, download=True, transform=transform_test, target_transform=target_transform)
else:
  c100_tt = cifar100CoarseTargetTransform()
  bd_labels = c100_tt.coarse2fine(options.backdoor_class)
  target_transform = CustomBDTT(bd_labels, options.target_class)
  p, m = options.backdoor_dataset.rsplit('.', 1)
  backdoor_dataset = import_from(p, m)(root='./data', train=options.use_train, download=True, transform=transform_test, target_transform=target_transform)
  print('fine labels of', options.backdoor_class, 'is:', bd_labels)
  print('target class:', options.target_class)
  dataset, _ = separate_class(backdoor_dataset, bd_labels)

# evaluate the model
dataloader = torch.utils.data.DataLoader(dataset, batch_size=options.batch, shuffle=False)
results = evaluate(model, dataloader, device)
if options.clean_eval:
  print('RESULTS:', results)
else:
  print('RESULTS:', results[:-1], results[-1][options.target_class])

