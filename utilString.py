#input: fileName string,
#output: corresponding labeled fileName string
#eg: given 'cs221_dataset/VGH_Training/0.1_280_5_5.jpg', 
#returns: 'cs221_dataset/VGH_Training/labeled/1_280_5_5.jpg'
def getLabeledName(fName):
  findString = 'Training/'
  i = fName.find(findString) + len(findString)
  labeledName = fName[0:i] + 'labeled/' + fName[i+2:]
  return labeledName

def getOutputName(fName):
  findString = 'Training/'
  i = fName.find(findString) + len(findString)
  outputName = 'output_files/' + fName[i+2:]
  return outputName

# ex input: '1.2.jpg' 
# output: '{1.2}.jpg'
def latexFileName(fName):
  return '{' + fName[0:-4] + '}' + fName[-4:]
