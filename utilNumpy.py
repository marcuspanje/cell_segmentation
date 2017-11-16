import json
from scipy import misc
import numpy as np 


with open('fileNames.json') as f:
  allNames = json.load(f)

n = allNames['train'][0]
im = misc.imread(n, mode:


