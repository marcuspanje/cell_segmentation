# Code in file nn/two_layer_net_nn.py
import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
          torch.nn.Linear(H, D_out),

# The nn package also contains definitions of popular loss functions; in this
loss_fn = torch.nn.MSELoss(size_average=False)
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # doing so you pass a Variable of input data to the Module and it produces

  # Compute and print loss. We pass Variables containing the predicted and true
  loss = loss_fn(y_pred, y)
  # Zero the gradients before running the backward pass.
  # Backward pass: compute gradient of the loss with respect to all the learnable
    # in Variables with requires_grad=True, so this call will compute gradients for
  # we can access its data and gradients like we did before.
    
