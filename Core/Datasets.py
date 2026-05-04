# Code processing data sample can be messy .
# training code + data loading + labels + shuffling = All mixed together  is hard to read and manipulate 
# The soluion is  two tools :
  # Datasets : The storage ,It stores all your data samples and stores their corresponding labels like library 
  # Dataloader : The  Delivery System. it wraps around the dataset  and delivers data in batched your model  
# We can index datasets manually like a list or use a dataloader to automatically handle batching, shuffling and loading the data in parallel using multiprocessing workers.

from torch.utils.data import Dataset,DataLoader 
from torchvision import datasets, transforms 

dataset = datasets.MNIST(
    root= "data",
    train = True,
    download=True,
    transform= transforms.ToTensor()
)

DataLoader = DataLoader(dataset, batch_size=5, shuffle=True)

for image,lable in DataLoader:
    print(image.shape)
    print(lable.shape)
    break
