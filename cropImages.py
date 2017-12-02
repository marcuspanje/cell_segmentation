from util import *
from scipy import misc
import torchvision.transforms as transforms

with open('fileNames.json', 'r') as f:
  allNames = json.load(f)

testNames = allNames['test']

for fname in testNames:
  print(fname)
  lname = getLabeledName(fname)
  im = load(fname)
  sz = im.shape
  labels = get_labels(lname)
  prefix = 'cropped_'  
  write_image_from_scores(labels, prefix + lname)
  misc.imsave(prefix + fname, transforms.ToPILImage().__call__(im))
  
  


