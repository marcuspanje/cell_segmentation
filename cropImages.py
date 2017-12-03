from util import *
from scipy import misc
import torchvision.transforms as transforms

with open('fileNames.json', 'r') as f:
  allNames = json.load(f)

allNames = allNames['test'] + allNames['validate'] + allNames['train']

for fname in allNames:
  print(fname)
  lname = getLabeledName(fname)
  im = load(fname)
  sz = im.shape
  labels = get_labels(lname)
  prefix = 'cropped_'  
  write_image_from_scores(labels, prefix + lname)
  misc.imsave(prefix + fname, transforms.ToPILImage().__call__(im))
  
  


