#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import math
from PIL import Image
import pdb
import json
from util import * 
from fcn32stats import FCN32stats

use_cuda = torch.cuda.is_available() 
print("use_cuda: {}".format(use_cuda))
dtype = torch.FloatTensor #if use_cuda else torch.FloatTensor

vgg16 = models.vgg16(True)
fcn = FCN32stats(3)
fcn.copy_params_from_vgg16(vgg16)

with open('fileNames.json', 'r') as f:
  allNames = json.load(f)


# N is batch size
# dim1 - horizontal dimension
# dim2 - vertical dimension
# num_chan - RGB dimension
dim1, dim2, num_chan= 224, 224, 3

num_train = 2
num_samples = num_train

train_ex = torch.FloatTensor(num_train, num_chan, dim2, dim1)

for i in range(num_train):
  filename = allNames['train'][i]
  train_ex[i] = load(filename, dtype)

if use_cuda:
  train_ex = train_ex.cuda()
  fcn = fcn.cuda()
    
print('learning starting')

inputs = Variable(train_ex)
outputs, outputScores = fcn.forward(inputs)


  





