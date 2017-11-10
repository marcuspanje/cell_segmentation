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
    self.classifier = None
    self.fcn = nn.Sequential(
      nn.Dropout(),
      nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(0.5, 0.5), padding=(1, 1))
    )

  def forward(self, x):
    x = self.features(x)
    #x = x.view(x.size(0), -1)
    x = self.fcn(x)
    return x
    
vgg = models.vgg16(True)
vggfcn = vggfcn(vgg)

with open('fileNames.json', 'r') as f:
  allNames = json.load(f)

fNames = allNames['train'][0:2]

for name in fNames:
  print('analyzing {}'.format(name))
  im = load(name, dtype)
  out = vggfcn.forward(im)
  print(out)
  #print out max score class
  #vmax,imax = torch.max(scores, 1)
  #print('results: {}, score: {}'.format(classNames[imax.data[0]], vmax.data[0]))



