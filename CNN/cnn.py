# This is the for the image things .
import torch 
import torch.nn as nn
# Kernal is the sliding window that does the actual workd. It is a small matrix of the number (weights) So when we define the kernal size as 2 then it means that (2*2) it multiply the its own internal weight by the pixel values it curenlt hovering it 
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Self represents the instance if the object itself. It acts as storage locker for the object. By saying  self.conv1 we are telling python save this cnn layers inside the specific object so that i can use it later in the forward function .
        # Learn features from the image 
        self.conv1 = nn.Conv2d(2,16,2) # 2 channels in, 16 filter out and 3*3 kernel 
        self.conv2 = nn.Conv2d(16,32,2) # 16 channel in, 32 filter out and 3*3 kernel
        self.fc = nn.Linear(32*6*6, 10) # 32 channels in, 10 output classes
# CNN layers works with the 3D cubes of the data and the final output is the linear layer.        
# This is forward function which is the convertor belt of the cnn where it takes an RAW data that is image and enters at one end end and goes through several transformation station and come out the other end as prediction .
        
    def forward(self,x):
        x = torch.relu(self.conv1(x)) # The image passes through the first set of 16 filters conv1   then ReLu activation function is applied , ReLu is the filter postivity , It replaces all negative number with the zero (f(x) = max(0,x)) without Relu the network is just be a giant linear equation makit it imposible to learn the complex pattern 
        x = torch.max_pool2d(x,2)    # Max pooling is a downsampling technique that reduces the spatial dimensions of the feature maps while retaining the most important information. It works by sliding a window (in this case, 2x2) across the input feature map and taking the maximum value within that window. This helps to reduce the computational load and also makes the model more robust to small translations in the input image.
        x = torch.relu(self.conv2(x)) # The output of the first convolutional layer is then passed through the second convolutional layer (conv2) which has 32 filters, and again ReLu activation function is applied to introduce non-linearity.
        x = torch.max_pool2d(x,2) # Again max pooling is applied to further reduce the spatial dimensions of the feature maps. After the second convolutional layer and max pooling, the feature maps are flattened into a 1D vector using x.view(x.size(0), -1). This is necessary because the fully connected layer (fc) expects a 1D input. Finally, the flattened feature vector is passed through the fully connected layer (fc) which produces the final output of the network, which is a vector of size 10 corresponding to the 10 output classes.
        x = x.view(x.size(0),-1) # This is flattening step ,it reshape the 3D cube of the data (Channels ,Height,Width) into a 1D long line of numbers . -1 means that it tell the pytorch to figure out how many numbers are left and put them in single row .
        x = self.fc(x) # Now after the caluclation of the features and flattening the data we pass it through the fully connected layer which is the decision making part of the network and it will give us the output of 10 classes
        return x 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SimpleCNN()
model.to(device)
print(model)



# In output we see the fc which means fully connected layer and this is the decision making part of the network 