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
batch_size = 2
dim1, dim2, num_chan= 224, 224, 3

num_train = len(allNames['train'])
num_validate = len(allNames['validate'])
num_batch = int(math.ceil(num_train/batch_size))

train_ex = torch.FloatTensor(num_train, num_chan, dim2, dim1).type(dtype)
label_ex = torch.LongTensor(num_train, dim2, dim1).type(torch.LongTensor)
valid_ex = torch.FloatTensor(num_validate, num_chan, dim2, dim1).type(dtype)
valid_lb = torch.LongTensor(num_validate, dim2, dim1).type(torch.LongTensor)


for i in range(len(allNames['train'])):
  filename = allNames['train'][i]
  train_ex[i] = load(filename, dtype)
  label_ex[i] = get_labels(getLabeledName(filename), torch.LongTensor)

for i in range(len(allNames['validate'])):
  filename = allNames['validate'][i]
  valid_ex[i] = load(filename, dtype)
  valid_lb[i] = get_labels(getLabeledName(filename), torch.LongTensor)
  
train_indices = np.arange(num_train)
np.random.shuffle(train_indices)

learning_rate = 3e-3
momentum = 0.9
epochs = 3

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(fcn.parameters() ,lr = learning_rate, momentum = 0.9)

if use_cuda:
  label_ex = label_ex.cuda()
  train_ex = train_ex.cuda()
  valid_ex = valid_ex.cuda()
  valid_lb = valid_lb.cuda()
  fcn = fcn.cuda()
  criterion.cuda()

print('learning starting')

for t in range(epochs):

  #make sure we iterate over the dataset once
  for i in range(num_batch):
    start_idx = (i*batch_size)%num_train
    end_idx = min(start_idx + batch_size, num_train-1)
    if end_idx == start_idx:
      break

    idx = torch.LongTensor(train_indices[start_idx:end_idx])
    if use_cuda:
      idx = idx.cuda()
    
    num_samples = len(idx)
    #print(train_ex[idx,:,:,:].size())
    #print(label_ex[idx,:,:,:].size())
    inputs = Variable(train_ex[idx,:,:,:])
    labelsND = Variable(label_ex[idx,:,:])
    labels = labelsND.view(num_samples * dim1 * dim2)
    outputsND = fcn.forward(inputs)
    outputs = fcn.forwardLoss(outputsND, num_samples, dim1, dim2, num_chan)

    #print(outputs.size())
    #print(label.size())
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #compute validation accuracy
    num_val_samps = 1
    val_input = Variable(valid_ex[0:num_val_samps,:,:,:])
    val_output = fcn(val_input)
    val_labels = torch.max(val_output.data,1)[1]

    #compute training accuracy


    acc = torch.sum(val_labels==valid_lb[0:num_val_samps,:,:])/(dim1*dim2*num_val_samps*1.0)

    train_out_labels = torch.max(outputs.data, 1)[1]
    train_acc = (torch.sum(train_out_labels == labels.data))/(len(labels))
    print('epoch: %d, iter:%d, loss: %.3f, accuracy: %.5f, train_accuracy: %.5f' % (t,i, loss.cpu().data[0],acc, train_acc))


torch.save(fcn.state_dict(), 'training.pt')

