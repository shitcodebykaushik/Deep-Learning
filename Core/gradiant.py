import torch 
import time 

x = torch.tensor(3.,requires_grad=True) 

y = x*x   # function of x

y.backward()

print("x.grad =", x.grad)