# GELU applies the Gaussian Error Linear Unit activation function to the input data. The GELU function is defined as: f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))). It is a smooth, non-linear activation function that combines the properties of both ReLU and sigmoid functions. GELU is often used in transformer models and has been shown to perform well in various natural language processing tasks. When you pass a tensor through a GELU layer, it will apply the GELU function element-wise to the input tensor, allowing the model to learn complex patterns in the data while maintaining computational efficiency.
import torch
import torch.nn as nn
gelu = nn.GELU()
input_tensor = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]])
output_tensor = gelu(input_tensor)
print("Input Tensor :\n", input_tensor)
print("\n")
print("Output Tensor :\n", output_tensor)