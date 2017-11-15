#various util functions
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np

#input: fileName string,
#output: corresponding labeled fileName string
#eg: given 'cs221_dataset/VGH_Training/0.1_280_5_5.jpg', 
#returns: 'cs221_dataset/VGH_Training/labeled/1_280_5_5.jpg'
def getLabeledName(fName):
  findString = 'Training/'
  i = fName.find(findString) + len(findString)
  labeledName = fName[0:i] + 'labeled/' + fName[i+2:]
  return labeledName


#input: filename of an image
#loads, transforms image for nnet 
#output: pytorch tensor 
def load(name, dtype=torch.FloatTensor):
  im = Image.open(name)
  
  loader = transforms.Compose([
    transforms.CenterCrop(min(im.size)), # make square
    transforms.Scale((224,224)),  # scale to VGG input size
    transforms.ToTensor()])

  im = loader(im).type(dtype)
  #im = im.unsqueeze(0)
  return im



#input: filename
#output: pytorch tensor with output variables
def get_labels(fn, dtype):
  label_im = load(fn, dtype).numpy()
  
  #Labeling scheme:
  #0 tumor cells, 1 non-tumor cells, 2 other
  isnotblack =  np.logical_or((label_im[0,:,:] > 50.0/256.0), (label_im[1,:,:] > 50.0/256.0))
  isblack = np.logical_not(isnotblack)
  isred = isnotblack * (label_im[0,:,:] > label_im[1,:,:])
  isgreen = isnotblack * (label_im[0,:,:] < label_im[1,:,:])
  label_arr =  0*isred + 1*isgreen + 2*isblack

  return torch.from_numpy(label_arr).type(dtype)
                                            
