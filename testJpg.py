from os.path import join, isfile
import numpy as np
from PIL import Image
from os import listdir

phases = ["train", "val", "test"]
for phase in phases:
	print (phase, ":")
	data_dir = "/home/DeeptiM/cs221/cell_segmentation/cs221_dataset/"
	label_path = join(data_dir + phase + '_labels.txt')
	#label_list = [line.strip() for line in open(label_path, 'r')]
	mypath = data_dir + "VGH_Training/labeled_preprocess/"
	label_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	for label_file in label_list:
		if label_file.split(".")[-1] == "png":
			data = Image.open(join(mypath, label_file))
			data_array = np.array(data, np.uint8)
			if np.max(data_array) > 2:
				print (label_file)
				print (data_array.shape, np.min(data_array), np.max(data_array))
			#assert np.min(data_array) >= 0
			#assert np.max(data_array) < 3
