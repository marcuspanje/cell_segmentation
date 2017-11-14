#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import math
#from ImageNetClassNames import classNames.type(dtype)
from PIL import Image
import pdb
import json
from util import * 
from fcn32s import FCN32s

use_cuda = torch.cuda.is_available() 
print("use_cuda: {}".format(use_cuda))
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    
vgg16 = models.vgg16(True)
fcn = FCN32s(3)
fcn.copy_params_from_vgg16(vgg16)

with open('fileNames.json', 'r') as f:
  allNames = json.load(f)


# N is batch size
# dim1 - horizontal dimension
# dim2 - vertical dimension
# num_chan - RGB dimension
'''
batch_size = 1
dim1, dim2, num_chan= 224, 224, 3

num_train = len(allNames['train'])
num_batch = int(math.ceil(num_train/batch_size))

train_ex = torch.FloatTensor(num_train, num_chan, dim2, dim1).type(dtype)
label_ex = torch.FloatTensor(num_train, num_chan, dim2, dim1).type(dtype)
for i in range(len(allNames['train'])):
  filename = allNames['train'][i]
  train_ex[i] = load(filename, dtype)
  label_ex[i] = get_labels(getLabeledName(filename), dtype)

train_indices = np.arange(num_train)
np.random.shuffle(train_indices)

learning_rate = 1e-4
momentum = 0.9
epochs = 5

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(vggfcn.parameters() ,lr = learning_rate, momentum = 0.9)
print('learning starting')

for t in range(epochs):

  #make sure we iterate over the dataset once
  for i in range(num_batch):
    start_idx = (i*batch_size)%num_train
    end_idx = min(start_idx + batch_size, num_train-1)
    idx = train_indices[start_idx:end_idx]
    inputs = Variable(train_ex[idx,:,:,:])
    labels = Variable(label_ex[idx,:,:,:])
    outputs = vggfcn(inputs)

    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch: %d, batch: %d, loss: %.3f' % (t,i,loss.data[0]))

'''


