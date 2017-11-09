#!/usr/bin/env python
import os
import random
from random import shuffle
from math import ceil
import json

#eg: given 'cs221_dataset/VGH_Training/0.1_280_5_5.jpg', 
#returns: 'cs221_dataset/VGH_Training/labeled/1_280_5_5.jpg'
def getLabeledName(fName):
  findString = 'Training/'
  i = fName.find(findString) + len(findString)
  labeledName = fName[0:i] + 'labeled/' + fName[i+2:]
  return labeledName

#get dictionary of train,test,validate file names

random.seed(10)
vgh = os.listdir('cs221_dataset/VGH_Training/')
vgh = vgh[0:-1]
shuffle(vgh) #remove labeled
vgh = ['cs221_dataset/VGH_Training/' + s for s in vgh]

nki = os.listdir('cs221_dataset/NKI_Training/')
nki = nki[0:-1]
shuffle(nki) #remove labeled 
nki = ['cs221_dataset/NKI_Training/' + s for s in nki]

percentage = (70, 20, 10) #train, test, validation

vghTrainLastIndex = round(ceil(percentage[0] * len(vgh) / 100))
vghTestLastIndex = round(vghTrainLastIndex + ceil(percentage[1]*len(vgh) / 100))

nkiTrainLastIndex = round(ceil(percentage[0] * len(nki) / 100))
nkiTestLastIndex = round(nkiTrainLastIndex + ceil(percentage[1]*len(nki) / 100))

train = vgh[0:vghTrainLastIndex] + nki[0:nkiTrainLastIndex]
shuffle(train)
test = vgh[vghTrainLastIndex:vghTestLastIndex] + nki[nkiTrainLastIndex:nkiTestLastIndex]
shuffle(test)
validate = vgh[vghTrainLastIndex:] + nki[nkiTestLastIndex:]
shuffle(validate)
fileNames = {'train': train, 'test': test, 'validate': validate}

with open('fileNames.json', 'w') as f:
  json.dump(fileNames, f, indent=1)
  

