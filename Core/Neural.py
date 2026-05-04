import torch 
import time
# Input (2*3)

A = torch.tensor ([[1,2,3],
                   [4,5,6]])

# Weights (3*2)
B = torch.tensor ([[1,2],
                   [3,4],
                   [5,6]])

# Output (2*2) 
C = torch.matmul(A, B)
print("Input A:\n", A)
print("Weights B:\n", B)
print("Output C (A @ B):\n", C)