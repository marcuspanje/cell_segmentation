import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import json
from util import *

with open('fileNames.json') as f:
  allNames = json.load(f)

files = allNames['train'][0:3]
for name in files:
  im = load(getLabeledName(name)).numpy()
  im = 
  if im
  
  

  
