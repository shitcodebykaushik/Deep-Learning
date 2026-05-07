# ReLU stands for the rectified linear unit activation function .
# ReLU applies the rectified linear unit function element-wise : f(x) = max(0,x) . It is a non linear activation function that is commonly used in neural networks to introduce non-linearity into the model . ReLU is computationally efficient and helps to mitigate the vanishing gradient problem that can occur with other activation functions like sigmoid or tanh . When you pass a tensor through a ReLU layer , it will set all negative values to zero and keep all positive values unchanged . This allows the model to learn complex patterns in the data while maintaining computational efficiency .
# It is the simplest activation function with the one rule :- If the number is negative then make it zero and if the number is positive then keep it as it is .
# Withou ReLu stacking 10 linear layers is mathematically the same as just one linear layer,they all collapse into a single matrix . ReLu breaks that by adding a bend in the line,forcing the network to actually learn complex patterns .
#  It is simple a gatekeeper: negative number gets blocker(set 0 ) possitive signal passes thoufh unchanged 
import torch 
import torch.nn as nn 

relu = nn.ReLU()
input_tensor = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]])
output_tensor = relu(input_tensor)
print("Input Tensor :\n", input_tensor)
print("\n")
print("Output Tensor :\n", output_tensor)