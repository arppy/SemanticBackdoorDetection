import argparse
import random
import numpy as np

import torch
import torchvision

from utils import import_from, training, robust_training, evaluate
from utils import database_statistics, cifar100CoarseTargetTransform, CustomMultiBDTT
from utils import parse_number_list, CustomTensorDataset


parser = argparse.ArgumentParser(description='Model Train')
parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
parser.add_argument('--data_seed', type=int, default=1234567890, help='dataset shuffle random seed')
parser.add_argument('--dataset', type=str, default='torchvision.datasets.CIFAR10', help='dataset name')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--backdoor_dataset', type=str, default=None, help='poisson dataset name (e.g. torchvision.datasets.CIFAR100)')
parser.add_argument('--backdoor_class', type=str, help='comma-separated list of backdoor classes (e.g., "13,5,6,...,10")')
parser.add_argument('--evaluate', default=False, action='store_true', help='evaluation mode')
parser.add_argument('--val_size', type=float, default=0.1, help='fraction of validation set')
parser.add_argument('--adversarial', default=False, action='store_true', help='adversarial model train')
parser.add_argument('--ddpm_path', type=str, default=None)
parser.add_argument('--ddpm_backdoor_path', type=str, default=None)
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
  backdoor_list = parse_number_list(options.backdoor_class)
  c100_tt = cifar100CoarseTargetTransform()
  bd_labels = []
  for target_class, backdoor_class in enumerate(backdoor_list):
    bd_labels.append(c100_tt.coarse2fine(backdoor_class))
  bd_labels_tensor = torch.stack(bd_labels)
  target_transform = CustomMultiBDTT(bd_labels_tensor)
  selected_backdoor_train = import_from('torchvision.datasets', 'CIFAR100')(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
  selected_backdoor_test = import_from('torchvision.datasets', 'CIFAR100')(root='./data', train=False, download=True, transform=transform_test, target_transform=target_transform)
  save_name += "-" + options.backdoor_class.replace(',', '-') + "-" + database_statistics[options.backdoor_dataset]['name']
  if options.adversarial :
    npzfile_backdoor = np.load(options.ddpm_backdoor_path)
    images_backdoor = npzfile_backdoor['image']
    labels_backdoor = npzfile_backdoor['label'].astype(int)
    selected_ddpm_train_backdoor = CustomTensorDataset(images_backdoor, labels_backdoor, transform=transform, target_transform=target_transform)

save_name += "_ds" + str(options.data_seed) + "_b" + str(options.batch)

p, m = options.dataset.rsplit('.', 1)
dataset_func = import_from(p, m)
trainset = dataset_func(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
testset = dataset_func(root='./data', train=False, download=True, transform=transform_test, target_transform=target_transform)
val_size = int(options.val_size*len(trainset))
trainset, valset = torch.utils.data.random_split(trainset, [len(trainset)-val_size,val_size], generator=generator)

list_of_trainset = [trainset]
list_of_testset = [testset]
weights = [10.0] * len(trainset)
print("len(trainset)",len(trainset),)
if options.backdoor_class is not None and options.backdoor_dataset != options.dataset and options.model_reference is None :
  weights += [18.0] * len(selected_backdoor_train)
  print("len(trainset_backdoor)", len(selected_backdoor_train), )
  list_of_trainset.append(selected_backdoor_train)
  list_of_testset.append(selected_backdoor_test)
  if options.adversarial is not None:
    print("len(trainset_ddpm_backdoor)",len(selected_ddpm_train_backdoor))
    weights += [2.0] * len(selected_ddpm_train_backdoor)
    list_of_trainset.append(selected_ddpm_train_backdoor)

sampler=None
if options.adversarial is not None :
  npzfile = np.load(options.ddpm_path)
  images_c10 = npzfile['image']
  labels_c10 = npzfile['label'].astype(int)
  ddpm_train = CustomTensorDataset(images_c10, labels_c10, transform=transform)
  weights += [1.0] * len(ddpm_train)
  print("len(trainset_ddpm)", len(ddpm_train), )
  list_of_trainset.append(ddpm_train)
  sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
  # weights are based on Gowal2020Uncovering Page 9, Figure 5: https://arxiv.org/pdf/2110.09468.pdf (~ 0.3 original, ~ 0.7 ddpm)

if len(list_of_trainset) > 1 :
  trainset = torch.utils.data.ConcatDataset(list_of_trainset)
if len(list_of_testset) > 1 :
  testset = torch.utils.data.ConcatDataset(list_of_testset)

max_samples_per_epoch = database_statistics[options.dataset]['samples_per_epoch']

if sampler is None :
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=options.batch, generator=generator, shuffle=True)
else :
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=options.batch, generator=generator, sampler=sampler)

val_loader = torch.utils.data.DataLoader(valset, batch_size=options.batch, shuffle=True, generator=generator)
if not options.evaluate:
  learning_rate = 0.1
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

