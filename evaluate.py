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
from fcn32s import FCN32s

use_cuda = torch.cuda.is_available() 
print("use_cuda: {}".format(use_cuda))
dtype = torch.FloatTensor #if use_cuda else torch.FloatTensor

vgg16 = models.vgg16(True)
fcn = FCN32s(3)
fcn.load_state_dict(torch.load('saved_models/trained_model_nov21.pth'))

with open('fileNames.json', 'r') as f:
  allNames = json.load(f)

testNames = allNames['test']

dim1, dim2, num_chan= 224, 224, 3
num_test = len(testNames)
test_ex = torch.FloatTensor(num_test, num_chan, dim2, dim1)

for i in range(len(testNames)):
  filename = testNames[i]
  test_ex[i] = load(filename)

if use_cuda:
  test_ex = test_ex.cuda()
  fcn = fcn.cuda()

print('applying network to test images')

test_ex = Variable(test_ex)
output_classes = fcn.forward(test_ex, num_test, dim1, dim2, num_chan)[1].data
output_classes = get_pixel_classes(output_classes)

for i in range(len(testNames)):
  fn = testNames[i]
  write_image_from_scores(output_classes[i].cpu(), getOutputName(fn))

  




