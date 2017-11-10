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
from util import getLabeledName

use_cuda = torch.cuda.is_available() 
print("use_cuda: {}".format(use_cuda))
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
vgg = models.vgg16(True)
if use_cuda:
  vgg.cuda()

class vggfcn(nn.Module):
  def __init__(self, vgg):
    super(vggfcn, self).__init__()
    
    self.features = vgg.features
    self.classifier = None
    self.fcn = nn.Sequential(
      nn.Dropout(),
      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(0.5, 0.5), padding=(1, 1))
    )

    def forward(self, x):
      x = x.features(x)
      x = x.view(x.size(0), -1)
      x = x.fcn(x)
      return x
    



path = 'stuff/'
fNames = ['cat1.jpeg', 'dog1.jpeg']
fNames = [path + name for name in fNames]


#f = open('fileNames.json', 'r')
#allNames = json.load(f)
#f.close()
#fNames = [allNames['train'][0], getLabeledName(allNames['test'][0])]

#https://github.com/alexis-jacq/Pytorch-Tutorials/blob/cd17b027b1c5daf19a94d586cd5c4c573d6c287e/Neural_Style.py#L26
def load(fileName):
  im = Image.open(name)
  
  loader = transforms.Compose([
    transforms.CenterCrop(min(im.size)), # make square
    transforms.Scale((224,224)),  # scale to VGG input size
    #transforms.Scale((224)),  # scale to VGG input size
    transforms.ToTensor()])
    
  im = Variable(loader(im)).type(dtype)
  if use_cuda:
    im.cuda()
  im = im.unsqueeze(0)
  return im

for name in fNames:
  print('analyzing {}'.format(name))
  im = load(name)
  scores = vgg.forward(im)
  #print out max score class
  vmax,imax = torch.max(scores, 1)
  print('results: {}, score: {}'.format(classNames[imax.data[0]], vmax.data[0]))



