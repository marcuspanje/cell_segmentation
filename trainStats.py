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
from fcn32stats import FCN32stats

use_cuda = torch.cuda.is_available() 
print("use_cuda: {}".format(use_cuda))
dtype = torch.FloatTensor #if use_cuda else torch.FloatTensor

vgg16 = models.vgg16(True)
fcn = FCN32stats(3)
fcn.copy_params_from_vgg16(vgg16)

with open('fileNames.json', 'r') as f:
  allNames = json.load(f)


# N is batch size
# dim1 - horizontal dimension
# dim2 - vertical dimension
# num_chan - RGB dimension
batch_size = 16
dim1, dim2, num_chan= 224, 224, 3

num_train = len(allNames['train'])
num_validate = len(allNames['validate'])
num_batch = int(math.ceil(num_train/batch_size))

train_ex = torch.FloatTensor(num_train, num_chan, dim2, dim1)
label_ex = torch.LongTensor(num_train, dim2, dim1).type(torch.LongTensor)
valid_ex = torch.FloatTensor(num_validate, num_chan, dim2, dim1)
valid_lb = torch.LongTensor(num_validate, dim2, dim1).type(torch.LongTensor)

for i in range(len(allNames['train'])):
  filename = allNames['train'][i]
  train_ex[i] = load(filename, dtype)
  label_ex[i] = get_labels(getLabeledName(filename), dtype).type(torch.LongTensor)

for i in range(len(allNames['validate'])):
  filename = allNames['validate'][i]
  valid_ex[i] = load(filename, dtype)
  valid_lb[i] = get_labels(getLabeledName(filename), dtype).type(torch.LongTensor)

            
train_indices = np.arange(num_train)
np.random.shuffle(train_indices)

learning_rate = 1e-4
momentum = 0.9
epochs = 100

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fcn.parameters() ,lr = learning_rate)

if use_cuda:
  label_ex = label_ex.cuda()
  train_ex = train_ex.cuda()
  valid_ex = valid_ex.cuda()
  valid_lb = valid_lb.cuda()
  fcn = fcn.cuda()
  criterion.cuda()
    

print('learning starting')

#acc_loss_file = open('acc_losses.txt', 'w')

for t in range(epochs):

  train_accuracies = []
  valid_accuracies = []
  losses = []
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
    inputs = Variable(train_ex[idx,:,:,:])
    labels = Variable(label_ex[idx,:,:].view(num_samples*dim1*dim2))
    outputs, outputScores = fcn.forward(inputs, num_samples, dim1, dim2, num_chan)

    #print('output')
    #print(outputs.data[0])
        
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #compute train accuracy, loss
    train_labels = torch.max(outputs.data, 1)[1].view(num_samples*dim1*dim2)
    train_acc = torch.sum(train_labels == labels.data)/len(labels)
    train_accuracies.append(train_acc)
    losses.append(loss.data[0])

    #deallocate memory 
    inputs = None
    train_labels = None
    outputs = None
    ouptutScores = None

    #compute validation accuracy
    num_val_samps = num_validate
    val_input = Variable(valid_ex[0:num_val_samps,:,:,:])
    val_output = fcn.forward_without_permute(val_input)
    val_labels = torch.max(val_output.data,1)[1]

    valid_acc = torch.sum(val_labels==valid_lb[0:num_val_samps,:,:])/(dim1*dim2*num_val_samps*1.0)
    #print('epoch: %d, batch: %d, loss: %.3f, accuracy: %.5f' % (t,i,loss.data[0],acc))

    valid_accuracies.append(valid_acc)

    #deallocate memory
    val_input = None
    val_output = None
    val_labels = None



  avg_batch_acc  = sum(valid_accuracies)/len(valid_accuracies)
  avg_batch_loss = sum(losses)/len(losses)
  avg_train_acc = sum(train_accuracies)/len(train_accuracies)
  #acc_loss_file.write("%d,%.3f,%.5f\n" % (t,avg_batch_loss,avg_batch_acc))  
  print('epoch: %d, loss: %.3f, valid accuracy: %.5f, train accuracy: %.5f' % (t,avg_batch_loss,avg_batch_acc, avg_train_acc))


torch.save(fcn.state_dict(), 'trained_model_stats.pth')


  





