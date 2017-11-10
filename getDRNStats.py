import numpy as np
from PIL import Image
import os
from os.path import exists, join



data_dir = '/home/DeeptiM/cell_segmentation/cs221_dataset/' 
image_path = join(data_dir + 'train_images.txt')
#assert exists(image_path)
print (image_path)
assert (exists(image_path))
image_list = [line.strip() for line in open(image_path, 'r')]
imgArray = np.zeros((len(image_list),720,1128,3))
print (imgArray.shape)
for i,imgFile in enumerate(image_list):
	imgFilePath = join(data_dir + imgFile)
	assert (exists(imgFilePath))
	img = Image.open(imgFilePath)
	imgArray[i,:,:,:] = img

imgMean = np.mean(imgArray, axis=(0,1,2))
imgStd = np.std(imgArray, axis=(0,1,2))

print (imgMean, imgStd)
