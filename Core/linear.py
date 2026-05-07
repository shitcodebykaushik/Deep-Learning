 # This is the module use to create the fully connected  layer 
 # It applies  a linear transformation to incoming data using weights and an optional bias .
 # Applies a linear transformation to the incoming data: y = xA^T + b 
 # Linear is just a fancy multiplier, you give is some numbers it multiplies them by weights it has learned and then adds a bias to it ,and it give you back with the new numbers .
 # nn.linear (4,2) this take 4 number in ,spit 2 number out 
 # It has two main parameters : in_features and out_features . In features is the number of input features and out features is the number of output features . So if you have a layer with in features 10 and out features 5 , it means that this layer will take in a vector of 10 numbers and output a vector of 5 numbers . The weights and bias are learnable parameters that the model adjusts during training to minimize the loss function .

# Here we took a vector of 10 numbers and squashed them down into a single value using some learned weigth and a bias .
# Linear layers  expectes floats(decimal numbers) as input and output . So if you give it integers it will automatically convert them to floats before processing . 
import torch 
import time 
import torch.nn as nn 
layers = nn.Linear(in_features=2,out_features=1)
# When we defined the layers nn.Linear then it automatically created the weights and the bias for us .
print(layers)
print("\n")
actual_input = torch.tensor([[1.0,2.0]])
# Passing it thorugh the linear layer
output = layers(actual_input)  # When we ran this line then performed a dot product plus an addition .
print("Input :", actual_input)
print ("\n")
print("Output :", output)
print("\n")
print("The weights :", layers.weight)
print("\n")
print("The bias :", layers.bias)

# During training: We keep everything in RAM because it's thousands of times faster than a DB.After training: We store the Weights in a file.After making a prediction: We store the Final Result in a DB for your app to use.