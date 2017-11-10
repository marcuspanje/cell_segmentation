#! /usr/bin/env python

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from PIL import Image
import pdb
from util import getLabeledName, load


upsample  = nn.ConvTranspose2d(1,1,kernel_size=(3,3))
upsample.weight.data = torch.FloatTensor
mat = torch.FloatTensor([[2,4],[9,11]])
mat.unsqueeze_(1)
up = upsample(mat)

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

