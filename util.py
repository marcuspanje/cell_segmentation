#various util functions
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from scipy import misc
import json

#input: fileName string,
#output: corresponding labeled fileName string
#eg: given 'cs221_dataset/VGH_Training/0.1_280_5_5.jpg', 
#returns: 'cs221_dataset/VGH_Training/labeled/1_280_5_5.jpg'
def getLabeledName(fName):
  findString = 'Training/'
  i = fName.find(findString) + len(findString)
  labeledName = fName[0:i] + 'labeled/' + fName[i+2:]
  return labeledName

def getOutputName(fName):
  findString = 'Training/'
  i = fName.find(findString) + len(findString)
  outputName = 'output_files/' + fName[i+2:]
  return outputName


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
def get_labels(fn, dtype=torch.FloatTensor):
  label_im = load(fn, dtype).numpy()
  
  #Labeling scheme:
  #0 tumor cells, 1 non-tumor cells, 2 other
  isnotblack =  np.logical_or((label_im[0,:,:] > 50.0/256.0), (label_im[1,:,:] > 50.0/256.0))
  isblack = np.logical_not(isnotblack)
  isred = isnotblack * (label_im[0,:,:] > label_im[1,:,:])
  isgreen = isnotblack * (label_im[0,:,:] < label_im[1,:,:])
  label_arr =  0*isred + 1*isgreen + 2*isblack

  return torch.from_numpy(label_arr).type(dtype)

#net_output is a B x C x H x W tensor, where
#B is batch size, and C is the number of classes
#returns a B x H x W, tensor containing the class index that has
#the highest score.
def get_pixel_classes(net_output):
  return torch.max(net_output, 1)[1]

#from a HxW tensor containing classes for each pixel,
#write a HxW jpeg image, where each class corresponds to a color
def write_image_from_scores(pixel_class, name):
  #first transpose
  #pixel_class = pixel_class.numpy().transpose()
  pixel_class = pixel_class.numpy()
  H,W = pixel_class.shape
  isRed = pixel_class == 0
  isGreen = pixel_class == 1
  isBlack = pixel_class == 2
  red = np.array([190, 0, 0])
  black = np.array([0,0,0])
  green = np.array([0,255,0])
  im = np.zeros((H,W,3))
  im[isRed] = red
  im[isGreen] = green
  im[isBlack] = black
  misc.imsave(name, im)

''' 
with open('fileNames.json') as f: 
  allNames = json.load(f)  
  
n = allNames['train'][0]
n = 'cs221_dataset/VGH_Training/labeled/3_268_8_2.jpg'
print(n)
im = load(n)

pixel_class = get_labels(n)
write_image_from_scores(pixel_class, 'out.jpg')
'''


  

  
