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
    self.deconv = nn.ConvTranspose2d(512, 3, kernel_size=44, stride=30, bias=False)

  def forward(self, x):
    x = self.features(x)
    x = self.deconv(x)
    return x
    
myvgg = models.vgg16(True)
myvggfcn = vggfcn(myvgg)

with open('fileNames.json', 'r') as f:
  allNames = json.load(f)

name = allNames['train'][0]

print('analyzing {}'.format(name))
im = load(name, dtype)
out = myvggfcn.forward(im)
print(out)


