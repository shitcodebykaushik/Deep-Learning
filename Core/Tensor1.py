import torch
data = [[1,2],[3,4]]
torch_tensor = torch.tensor(data)   # Here wwe are using the tensor which changes the capabilites of the data and can do the gpu work and also the gradient work which is used in the backpropagation in the neural networks
print("Data:\n", data)   # This will print the data normal data  which is the list in the python 
print("\n")
print("Torch Tensor:\n", torch_tensor)  # GPU accelerated tensor 

# Shape is tuple of tensor dimensions .It simply determines the size of the  each dimension of tensor .
shape = (3,4) # 3 rows and 4 columns
# Shape tells the function How big to make the tensor i.e how many rows and columns or deeper dimensions it should have 
# Tuple means its a collection of number in () that describe how big each dimension is  
 # (3,4) means 3 rows and 4 columns
  # (3,4,5) means 3 rows and 4 columns and 5 depth
  
# Dimensionality of output tesnsor meaans the shape controls the size of the tensor gets creaated 
c = torch.zeros((2,3))  # This will create a tensor of zeros with the shape of (2,3) which means 2 rows and 3 columns
print("Tensor of zeros:\n",c)  # This will create a tensor of zeros with the shape of (2,3) which means 2 rows and 3 columns


# Tensor attributes describe their shape,datatpye and the device on which they  are stored 
tensor = torch.rand((3,4))  # This will create a tensor of random values with the shape of (3,4) which means 3 rows and 4 columnsj

print(tensor)
print(f"Shape:{tensor.shape}")
print(f"Datatype:{tensor.dtype}")
print(f"Device:{tensor.device}")


# Operation on tensors => There are  around  1200+ operations which we can do in the tensors 
# By default tensors are created ont the cpu but we can move them to the gpu  usine  .to 
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print(f"Device after moving to GPU:{tensor.device}")

print(f"The current device is {tensor.device}")
print(tensor)


# Tensor indexing and slicing 

t = torch.ones(2,2)
print("Tensor t:\n", t)
print("First row of t:", t[0])  # This will print the first row
print("First column of t:", t[:,0])  # This will print the first column
print("Last column of t:", t[:,-1])  # This will print the last

# Dimension means the number of the directions you need to navigate to reach a value in the the data 
# Joining tensors 
# torch.cat() is used to concatenate tensors along a specified dimension

# In-place operations 
# In-place operations are operations that modify the data of a tensor without making a copy. They are denoted by an underscore at the end of the function name, such as .add_() or .mul_(). In-place operations can be more memory efficient but should be used with caution as they can lead to unintended side effects if the original tensor is shared across different parts of the code.

t = torch.tensor([1,3,5])
print("Oringinal tensor t:",t)
t.add_(10)  # This will add 10 to each element of the tensor t in-place
print("Tensor t after in-place addition:",t)
