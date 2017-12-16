import itertools
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from util import *
from u_net import UNet
from collections import OrderedDict

use_cuda = torch.cuda.is_available()
print("use_cuda: {}".format(use_cuda))
dtype = torch.FloatTensor #if use_cuda else torch.FloatTensor

net = UNet(3)
state_dict = torch.load('saved_models/trained_model_unet_halved.pth')
dim1, dim2, num_chan= 224, 224, 3

# load params
net.load_state_dict(state_dict)

with open('fileNames.json', 'r') as f:
      allNames = json.load(f)

num_test = len(allNames['test'])
test_ex = torch.FloatTensor(num_test, num_chan, dim2, dim1)
label_ex = torch.LongTensor(num_test, dim2, dim1).type(torch.LongTensor)
testNames = allNames['test']
      
for i in range(len(testNames)):
    filename = testNames[i]
    test_ex[i] = load(filename)
    label_ex[i] = get_labels(getLabeledName(filename), dtype).type(torch.LongTensor)

if use_cuda:
    test_ex = test_ex.cuda()
    label_ex = label_ex.cuda()
    net = net.cuda()

print('applying network to test images')
    
test_ex = Variable(test_ex)
output_classes = net.forward(test_ex)[1].data
output = get_pixel_classes(output_classes).cpu().numpy()

test_ex = None
output_classes = None
labels = Variable(label_ex).data.cpu().numpy()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm,3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, cm[i, j], color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confmat.png')
            

print(output.shape)
print(labels.shape)
# Compute confusion matrix
cnf_matrix = confusion_matrix(np.ndarray.flatten(output), np.ndarray.flatten(labels))
np.set_printoptions(precision=5)
class_names = ['tumor','non-tumor','background']

# Plot Confusion Matrix
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title = "UNet")
