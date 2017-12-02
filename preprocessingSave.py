#import tensorflow as tf
import torch
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from os.path import join, exists
from scipy import misc
import random


#Change this to your local directory
data_dir = "/home/DeeptiM/cs221/cell_segmentation/cs221_dataset/"
phase = "val"
list_path = join(data_dir, phase + "_labels_orig.txt")

assert exists(list_path)

# read file path names from file
if exists(list_path):
  label_list = [line.strip() for line in open(list_path, 'r')]

for label_file in label_list:
  #label_img = Image.open(join(data_dir, label_file))
  #jimg_array = np.zeros((720, 1128, 3))
  #img_array[:,:,:] = label_img
  img_array = misc.imread(data_dir + label_file)
  # process array
  isnotblack =  np.logical_or((img_array[:,:,0] > 50), (img_array[:,:,1] > 50))
  isblack = np.logical_not(isnotblack)
  isred = isnotblack * (img_array[:,:,0] > img_array[:,:,1])
  isgreen = isnotblack * (img_array[:,:,0] < img_array[:,:,1])
  new_array =  0*isred + 1*isgreen + 2*isblack
  file_parts = label_file.split("/")
  part_label_file = "/".join(file_parts[:-2])
  extension = file_parts[-1].split(".")
  new_label_file = data_dir + part_label_file + "/labeled_preprocess/" + extension[0] + ".png"
  #save without normalizing pixel values
  #new_label_file = new_label_file[0:-3] + 'png'
  misc.toimage(new_array, cmin=0, cmax=255).save(new_label_file)
  #print (new_label_file)

print (phase + " done")
