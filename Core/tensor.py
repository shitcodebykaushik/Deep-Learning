import torch 

# Scalar (0D)
a = torch.tensor(5)

# Vector (1D)
b = torch.tensor([1, 2, 3])

# Matrix (2D)
c = torch.tensor([[1,2], [3,4]])

print("Scalar (0D):", a)
print("The dimensionality of a:", a.dim())
print("The shape of a:", a.shape)
print("\n")  # Just for better readability
print("Vector (1D):", b)
print("The dimensionality of b:", b.dim())
print("The shape of b:", b.shape)
print("Matrix (2D):", c)
print("The dimensionality of c:", c.dim())
print("The shape of c:", c.shape)



# All zeros 
x = torch.ones(3, 4)
print("Tensor of ones:\n", x)

# Models start with random weights, so we can create a tensor of random values
y = torch.rand(3, 4)
print("Tensor of random values:\n", y)


 # basic opertations 
 
t = torch.tensor([[1, 2], [3, 4]])
s = torch.tensor([[5, 6], [7, 8]])
print("Tensor t:\n", t)
print("Tensor s:\n", s)
print("\n") 
print(t*s)
print("\n") 



# Move the tensor to gpu directly as we have the gpu 

y1 = torch.tensor([[1, 2], [3, 4]])
i = y1.to("cuda")
y3 = torch.tensor([[5, 6], [7, 8]])
print("Tensor y on GPU:\n", i)

print ("\n")
# print(i+y3) # this will throw an error because y3 is on CPU and i is on GPU, we need to move y3 to GPU as well
print(i + y3.to("cuda"))  # Move y3 to GPU and then add them together  

