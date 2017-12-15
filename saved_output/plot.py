import numpy as np
import matplotlib as plt

fname = 'slurm-103288-model-withstats.out'
train_acc = []
valid_acc = []
with open(fname) as f:
  for line in f:
    arr = line.split()
    if arr[0] == 'epoch:':
      train_acc.append(float(arr[6][:-1]))
      valid_acc.append(float(arr[9][:-1]))
        
print('train')
print(train_acc)
print('valid')
print(valid_acc)


