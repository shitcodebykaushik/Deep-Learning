# It is PyTorch's toolbox that contains everthing you need build a neural network 

import torch.nn as nn 
# nn.module  is the base class 
# The Parent of everthing . Every layer,every network , is a child of nn.module
# nn.Module  = Blueprint of a building 
# Your Network = Actual building from that blueprint 
# Every individual layer is also module 
#nn.Linear()
#nn.Conv2d()
#nn.ReLU()
#nn.Dropout()

# Each one is a module  that does ONE specific operation on data 

# Network = Module of Modules 
# A neural network is just modules stacked inside modules 
# You can even put network inside networks 
# Every nn.module subclass implements the operations on input data in the forward method .
# We create an instances of NeuralNetwork and move it to the device and print its structure 

import torch 

class Mynetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define layers (Each is also a module)
        self.layer1 = nn.Linear(784,256)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(256,10)  # Hidden => output 
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 
    
model = Mynetwork()
print(model)

# nn.flatten = A module whoes job is reshape data . It converts multi-dimensional data into a !D line 
import torch.nn as nn 
flatten = nn.Flatten()
image = torch.rand((1,28,28))  # This is a random image with the shape of (1,28,28) which means 1 channel and 28 rows and 28 columns
flattened_image = flatten(image)  # This will flatten the image into a 1D
print("Original image shape:", image.shape)  # This will print the original shape of the image which is (1,28,28)
print("Flattened image shape:", flattened_image.shape)  # This will print the flattened shape
# nn.Linear = A module whoes job is learn patterns 
# nn.ReLU = A module whoes job is add non-linearity to the model and add complexity 
# nn.Softmax =  A module whoes job is convert the output into the final probalities 

# nn.linear = A module that learns pattens in the fully connected layers of the network . It  applies a mathematics transformation to learn pattens from the data 
# Output = (input * weight) + bias 

linear =  nn.Linear(in_features=784,out_features=256)

input = torch.rand(1,784)
output = linear(input)
print("Input shape:", input.shape)  # This will print the shape of the input which is (1,784)
print("Output shape:", output.shape)  # This will print the shape of the output which


# nn.ReLU = A module that adds non-lineraity to help network learn more comlex patters 

relu = nn.ReLU()
input = torch.tensor([-1.0,0.0,1.0])
output = relu(input)
print(output)