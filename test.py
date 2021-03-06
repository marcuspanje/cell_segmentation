#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import math
from ImageNetClassNames import classNames
from PIL import Image
import pdb
import json
from util import getLabeledName, load

use_cuda = torch.cuda.is_available() 
print("use_cuda: {}".format(use_cuda))
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class vggfcn(nn.Module):
  def __init__(self, vgg):
    super(vggfcn, self).__init__()
    
    self.features = vgg.features
    #self.classifier = vgg.classifier
    #ref: https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
    '''
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, 3, padding=100),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2

        # conv2
      nn.Conv2d(64, 128, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4

        # conv3
      nn.Conv2d(128, 256, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8

        # conv4
      nn.Conv2d(256, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16

        # conv5
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, 3, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/32
    '''

    self.deconv = nn.Sequential(
      nn.Conv2d(512, 4096, kernel_size=7),
      nn.ReLU(inplace=True),
      nn.Dropout2d(),
      nn.Conv2d(4096, 4096, 1),
      nn.ReLU(inplace=True),
      nn.Dropout2d(),
      nn.Conv2d(4096, 3, 1),
      nn.ConvTranspose2d(3, 3, 64, stride=32, bias=False)
    )


  def forward(self, x):
    x = self.features(x)
    print (x.size())
    x = self.deconv(x)
    x = x[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
    return x
    
vgg = models.vgg16(True)
vggfcn = vggfcn(vgg)

with open('fileNames.json', 'r') as f:
  allNames = json.load(f)

name = allNames['train'][0]

print('analyzing {}'.format(name))
im = load(name, dtype)
im = Variable(im)
im = im.unsqueeze(0)
out = vggfcn.forward(im)
print(out)

