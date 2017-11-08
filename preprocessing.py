#import tensorflow as tf
import torch
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from scipy import misc
import random
%matplotlib inline

# Reading data and cropping into images
# NKI data

#Change this to your local directory
os.chdir("C:\\Users\\Shiv\\Documents\\cs221\\proj")
print(os.getcwd())
cwd = os.getcwd()
ctr = 0
num_NKI_images = 107
NKI_full_x = np.zeros((num_NKI_images, 720, 1128, 3))
NKI_full_y = np.zeros((num_NKI_images, 720, 1128, 3))
os.chdir(cwd + "\\NKI_Training")
root = cwd + "\\NKI_Training\\labeled"

for file in glob.glob("0.*"):
  img = Image.open(file)
  labfile = os.path.join(root, file[2:])
  labimg = Image.open(labfile)
  NKI_full_x[ctr, :, :, :] = img
  NKI_full_y[ctr, :, :, :] = labimg
  ctr += 1

# VGH Data
ctr = 0
num_VGH_images = 51
VGH_full_x = np.zeros((num_VGH_images, 720, 1128, 3))
VGH_full_y = np.zeros((num_VGH_images, 720, 1128, 3))
os.chdir(cwd+"\\VGH_Training")
root = cwd+"\\VGH_Training\\labeled"

for file in glob.glob("0.*"):
  img = Image.open(file)
  labfile = os.path.join(root, file[2:])
  labimg = Image.open(labfile)
  VGH_full_x[ctr, :, :, :] = img
  VGH_full_y[ctr, :, :, :] = labimg
  ctr += 1

os.chdir(cwd)

NKI_labels = np.zeros((num_NKI_images,720,1128))
VGH_labels = np.zeros((num_VGH_images,720,1128))

#Labeling scheme:
#0 tumor cells, 1 non-tumor cells, 2 other

for f in range(0,num_NKI_images):
    isnotblack =  np.logical_or((NKI_full_y[f,:,:,0] > 50), (NKI_full_y[f,:,:,1] > 50))
    isblack = np.logical_not(isnotblack)
    isred = isnotblack * (NKI_full_y[f,:,:,0] > NKI_full_y[f,:,:,1])
    isgreen = isnotblack * (NKI_full_y[f,:,:,0] < NKI_full_y[f,:,:,1])
    NKI_labels[f,:,:] =  0*isred + 1*isgreen + 2*isblack

for f in range(0,num_VGH_images):
    isnotblack =  np.logical_or((VGH_full_y[f,:,:,0] > 50), (VGH_full_y[f,:,:,1] > 50))
    isblack = np.logical_not(isnotblack)
    isred = isnotblack * (VGH_full_y[f,:,:,0] > VGH_full_y[f,:,:,1])
    isgreen = isnotblack * (VGH_full_y[f,:,:,0] < VGH_full_y[f,:,:,1])
    VGH_labels[f,:,:] =  0*isred + 1*isgreen + 2*isblack



