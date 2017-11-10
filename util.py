#various util functions
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

#input: fileName string,
#output: corresponding labeled fileName string
#eg: given 'cs221_dataset/VGH_Training/0.1_280_5_5.jpg', 
#returns: 'cs221_dataset/VGH_Training/labeled/1_280_5_5.jpg'
def getLabeledName(fName):
  findString = 'Training/'
  i = fName.find(findString) + len(findString)
  labeledName = fName[0:i] + 'labeled/' + fName[i+2:]
  return labeledName


#input: filename of an image
#loads, transforms image for nnet 
#output: pytorch tensor 
def load(name, dtype):
  im = Image.open(name)
  
  loader = transforms.Compose([
    transforms.CenterCrop(min(im.size)), # make square
    transforms.Scale((224,224)),  # scale to VGG input size
    transforms.ToTensor()])

  im = Variable(loader(im)).type(dtype)
  im = im.unsqueeze(0)
  return im


