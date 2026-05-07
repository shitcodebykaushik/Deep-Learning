# nn.Sigmoid() it squishes any number into a value between 0 and 1 . No matter how big or small the input  is ,output is always between 0 and 1 . 
# It is a non linear activation function that is commonly used in neural networks to introduce non-linearity into the model . Sigmoid is computationally efficient and helps to mitigate the vanishing gradient problem that can occur with other activation functions like ReLU or tanh . When you pass a tensor through a sigmoid layer , it will apply the sigmoid function element-wise to the input tensor, allowing the model to learn complex patterns in the data while maintaining computational efficiency .
# It is a smooth curve that maps any real-valued number into the (0,1) interval . It is often used in the output layer of binary classification problems to represent probabilities . The sigmoid function is defined as: f(x) = 1 / (1 + exp(-x)) . When you pass a tensor through a sigmoid layer, it will apply the sigmoid function element-wise to the input tensor, allowing the model to learn complex patterns in the data while maintaining computational efficiency .
# It is a probabilistic machine : feed it any  number and it tells you how confident the model is that the answer is yes .
 
import torch 
import torch.nn as nn 
sigmoid = nn.Sigmoid()

x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]])
output = sigmoid(x)
print("Input Tensor :\n", x)
print("\n")
print("Output Tensor :\n", output)

