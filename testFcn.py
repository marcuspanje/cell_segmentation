#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import math
#from ImageNetClassNames import classNames.type(dtype)
from PIL import Image
import pdb
import json
from util import * 
from fcn32s import FCN32s

use_cuda = torch.cuda.is_available() 
print("use_cuda: {}".format(use_cuda))
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    
vgg16 = models.vgg16(True)
fcn = FCN32s(3)
fcn.copy_params_from_vgg16(vgg16)

with open('fileNames.json', 'r') as f:
  allNames = json.load(f)

name = allNames['train'][0]

