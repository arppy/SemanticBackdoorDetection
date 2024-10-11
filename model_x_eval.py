import os
import sys
import argparse
import random
import numpy as np

import torch
import torchvision.transforms as T
import torchvision.models as models
import torchvision.datasets as datasets

from utils import import_from, separate_class, identity, cross_evaluate, \
  Cutout, CustomTensorDataset, RandomDataset, GeneratedDataset, \
  database_statistics, MODEL_ARCHITECTURES, DATABASES, imagewoof, imagenette, \
  CustomClassLabelByIndex, DATABASE_SUBSET, ModelTransformWrapper

#from models.preact_resnet import PreActResNet18
#from models.resnetmod_ulp import resnet18_mod

parser = argparse.ArgumentParser(description='Model Cross Eval')
parser.add_argument('--seed', type=int, default=1234567890, help='random seed')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default='torchvision.datasets.CIFAR10', help='torch dataset name')
parser.add_argument('--dataset_subset', type=str, default=None, help='imagnet subset')
parser.add_argument('--dataset_dir', type=str, default=None, help='location of data directory')
parser.add_argument('--label', type=int, default=-1, help='class to be tested')
parser.add_argument('--a_model', type=str, default=None, help='model a')
parser.add_argument('--b_model', type=str, default=None, help='model b')
parser.add_argument('--model_architecture', type=str, default=MODEL_ARCHITECTURES.RESNET18.value, choices=[e.value for e in MODEL_ARCHITECTURES], help='load mode weights')
parser.add_argument('--model_wrapped', default=False, action='store_true')
parser.add_argument('--metric', type=str, default='torch.nn.functional.cross_entropy', help='metric')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--use_train', action='store_true', default=False, help='use training set')
parser.add_argument('--cutout', type=int, default=None)
parser.add_argument('--ddpm_path', type=str, default=None)
parser.add_argument('--adversarial', default=False, action='store_true', help='Adversarial model eval')
parser.add_argument('--list_of_backdoor_classes_that_need_to_avoid', type=int, nargs="*", default=None)
parser.add_argument('--random_data', type=int, nargs='+', default=None, help='use random dataset')
parser.add_argument('--generated_data', type=str, default=None, help='path of generated datasets')
parser.add_argument('--merge', action='store_true', default=False, help='merge models and evaluate')

options = parser.parse_args()
print('OPTIONS:', options)
device = torch.device('cuda:' + str(options.gpu) if torch.cuda.is_available() else 'cpu')
print('device:', device)
torch.backends.cudnn.deterministic = True

torch.manual_seed(options.seed)
random.seed(options.seed)
np.random.seed(options.seed)


num_classes = 10
if options.dataset == DATABASES.TINYIMAGENET.value :
  num_classes = database_statistics[options.dataset]['num_classes']
eps = None
step_size = None
steps = None
if options.adversarial:
  eps = 255.0 / 255.0
  steps = 10
  step_size = 0.01
  #if options.dataset == DATABASES.CIFAR10.value or options.dataset == DATABASES.SVHN.value :
  #  eps = 8.0 / 255.0
  #  step_size = 2.0 / 255.0
  #  steps = 10
  #else :
  #  eps = 0.0156862745
  #  step_size = 0.01
  #  steps = 3

mean = database_statistics[options.dataset]['mean']
std = database_statistics[options.dataset]['std']

if options.model_architecture == MODEL_ARCHITECTURES.WIDERESNET.value :
  #DMWideResNet = import_from('robustbench.model_zoo.architectures.dm_wide_resnet', 'DMWideResNet')
  #Swish = import_from('robustbench.model_zoo.architectures.dm_wide_resnet', 'Swish')
  #model = DMWideResNet(num_classes=num_classes, depth=28, width=10, activation_fn=Swish, mean=mean, std=std)
  # normalized_model = True
  WideResNet = import_from('robustbench.model_zoo.architectures.wide_resnet', 'WideResNet')
  model_a = WideResNet().to(device)
  model_b = WideResNet().to(device)
elif options.model_architecture == MODEL_ARCHITECTURES.PREACTRESNET18.value :
  model_a = PreActResNet18(num_classes)
  model_b = PreActResNet18(num_classes)
elif options.model_architecture == MODEL_ARCHITECTURES.ULP_RESNET_MOD.value :
  model_a = resnet18_mod(num_classes=num_classes)
  model_b = resnet18_mod(num_classes=num_classes)
  if options.model_wrapped:
    model_a = ModelTransformWrapper(model_a, T.Normalize(mean, std), device)
    model_b = ModelTransformWrapper(model_b, T.Normalize(mean, std), device)
else :
  if options.dataset == DATABASES.CIFAR10.value or options.dataset == DATABASES.SVHN.value :
    ResNet = import_from('robustbench.model_zoo.architectures.resnet', 'ResNet')
    BasicBlock = import_from('robustbench.model_zoo.architectures.resnet', 'BasicBlock')
    layers = [2, 2, 2, 2]
    model_a = ResNet(BasicBlock, layers, num_classes).to(device)
    model_b = ResNet(BasicBlock, layers, num_classes).to(device)
  else :
    model_a = models.resnet18(weights=None)
    model_a.fc = torch.nn.Linear(512, num_classes)
    model_a = model_a.to(device)
    model_b = models.resnet18(weights=None)
    model_b.fc = torch.nn.Linear(512, num_classes)
    model_b = model_b.to(device)

if options.a_model[-5] == '_1.pt' :
  load_file_a = torch.load(options.a_model)
  model_a.load_state_dict(load_file_a['model'])
  model_a = model_a.to(device)
else:
  model_a.load_state_dict(torch.load(options.a_model, map_location=device))

if options.b_model[-5] == '_1.pt' :
  load_file_b = torch.load(options.b_model)
  model_b.load_state_dict(load_file_b['model'])
  model_b = model_b.to(device)
else:
  model_b.load_state_dict(torch.load(options.b_model, map_location=device))

if options.model_wrapped :
  model_a = model_a.model
  model_b = model_b.model

transform_list = []
if options.dataset == DATABASES.IMAGENET.value :
  transform_list.append(T.Resize(256))
  transform_list.append(T.CenterCrop(224))
elif options.dataset == DATABASES.TINYIMAGENET.value:
  transform_list.append(T.CenterCrop(56))
elif options.dataset == DATABASES.AFHQ.value:
  transform_list.append(T.Resize(224))
if options.cutout is not None:
  if options.dataset == DATABASES.CIFAR10.value or options.dataset == DATABASES.SVHN.value :
    transform_list.append(T.RandomCrop(32, padding=4))
  transform_list.append(Cutout(options.cutout))
transform_list.append(T.ToTensor())
if eps is None :
  transform_list.append(T.Normalize(mean, std))
  transformNorm = None
else :
  transformNorm = T.Normalize(mean, std)
transform = T.Compose(transform_list)

if options.ddpm_path is not None :
  npzfile = np.load(options.ddpm_path)
  images = npzfile['image']
  labels = npzfile['label'].astype(int)
  dataset = CustomTensorDataset(images, labels, transform=transform)
  if options.list_of_backdoor_classes_that_need_to_avoid is not None :
    clean_ddpm = []
    for i in range(len(labels)):
      if labels[i] not in options.list_of_backdoor_classes_that_need_to_avoid:
        clean_ddpm.append(i)
    dataset = torch.utils.data.Subset(dataset, clean_ddpm)
else :
  if options.random_data is not None:
    dataset = RandomDataset(options.random_data)
  elif options.generated_data is not None:
    dataset = GeneratedDataset(os.path.join(options.generated_data,os.path.basename(options.a_model)), os.path.join(options.generated_data,os.path.basename(options.b_model)), transform=transform)
    #dataset = GeneratedDataset(os.path.join(options.generated_data,os.path.basename(options.a_model)+'_'+os.path.basename(options.b_model)), os.path.join(options.generated_data,os.path.basename(options.b_model)+'_'+os.path.basename(options.a_model)), transform=transform)
    #dataset = GeneratedDataset(os.path.join(options.generated_data,os.path.basename(options.a_model)), None, transform=transform)
    #dataset = GeneratedDataset(os.path.join(options.generated_data,os.path.basename(options.a_model)+'_'+os.path.basename(options.b_model)), None, transform=transform)
  else:
    if options.dataset == DATABASES.IMAGENET.value or options.dataset == DATABASES.TINYIMAGENET.value :
      if options.dataset_subset == DATABASE_SUBSET.IMAGEWOOF.value:
        data_scope = imagewoof
      elif options.dataset_subset == DATABASE_SUBSET.IMAGENETTE.value:
        data_scope = imagenette
      else:
        data_scope = []
      if len(data_scope) > 0:
        target_transform = CustomClassLabelByIndex(data_scope)
      else:
        target_transform = None
      dataset = datasets.ImageFolder(options.dataset_dir, transform=transform, target_transform=target_transform)
      if len(data_scope) > 0:
        dataset, _ = separate_class(dataset, data_scope)
    else:
      p, m = options.dataset.rsplit('.', 1)
      dataset_func = import_from(p, m)
      if options.dataset == DATABASES.SVHN.value :
        split = "train" if options.use_train else "test"
        dataset = dataset_func(root='./data', split=split, download=True, transform=transform)
      else :
        dataset = dataset_func(root='./data', train=options.use_train, download=True, transform=transform)
  if options.label > -1 :
    dataset, rest_data = separate_class(dataset, options.label)
  if options.list_of_backdoor_classes_that_need_to_avoid is not None:
    clean_images_train = []
    for i in range(len(dataset.targets)):
      if dataset.targets[i] not in options.list_of_backdoor_classes_that_need_to_avoid:
        clean_images_train.append(i)
    dataset = torch.utils.data.Subset(dataset, clean_images_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=options.batch, shuffle=False)

p, m = options.metric.rsplit('.', 1)
func_a = identity
func_b = identity
if m == 'cross_entropy':
  func_a = identity
  func_b = torch.nn.functional.softmax
elif m == 'kl_div':
  func_a = torch.nn.functional.log_softmax
  func_b = torch.nn.functional.softmax
elif m == 'argmax_dist':
  func_a = identity
  func_b = identity
elif m == 'cos_dist':
  func_a = torch.nn.functional.softmax
  func_b = torch.nn.functional.softmax
elif m == 'cos_dist_logit':
  func_a = identity
  func_b = identity
  m = 'cos_dist'
else:
  print('Not supported metric:', options.metric)
  sys.exit(1)

loss = import_from(p, m)
result = cross_evaluate(model_a, model_b, dataloader, device, loss, func_a, func_b, merge=options.merge, eps=eps,
                        step_size=step_size, steps=steps, transform=transformNorm)
torch.set_printoptions(linewidth=200)
print('RESULT:', result)

