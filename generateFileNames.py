#!/usr/bin/env python
import os
import random
from random import shuffle
from math import ceil
import json

#get dictionary of train,test,validate file names

random.seed(10)
vgh = os.listdir('cs221_dataset/VGH_Training/')
vgh.sort()
vgh = ['cs221_dataset/VGH_Training/' + s for s in vgh if s[-3:] == 'jpg']
shuffle(vgh) 

nki = os.listdir('cs221_dataset/NKI_Training/')
nki.sort()
nki = ['cs221_dataset/NKI_Training/' + s for s in nki if s[-3:] == 'jpg']
shuffle(nki)

percentage = (70, 20, 10) #train, test, validation

vghTrainLastIndex = round(ceil(percentage[0] * len(vgh) / 100))
vghTestLastIndex = round(vghTrainLastIndex + ceil(percentage[1]*len(vgh) / 100))

nkiTrainLastIndex = round(ceil(percentage[0] * len(nki) / 100))
nkiTestLastIndex = round(nkiTrainLastIndex + ceil(percentage[1]*len(nki) / 100))

train = vgh[0:vghTrainLastIndex] + nki[0:nkiTrainLastIndex]
shuffle(train)
test = vgh[vghTrainLastIndex:vghTestLastIndex] + nki[nkiTrainLastIndex:nkiTestLastIndex]
shuffle(test)
validate = vgh[vghTestLastIndex:] + nki[nkiTestLastIndex:]
shuffle(validate)
fileNames = {'train': train, 'test': test, 'validate': validate}

with open('fileNames.json', 'w') as f:
  json.dump(fileNames, f, indent=1, sort_keys=True)
  

