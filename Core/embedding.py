"""A neural network only understand numbers,"""
"""In pytorch an embedding is just a look up table. a matrix of shape(vocab_size, embdding_dim)"""
"""Each row is one words vector. You give it a word's integer index ,it returns that row."""

import torch 
import torch.nn as nn 

# Vocab 
vocab = {"<pad>":0, "hello":1, "world":2,"good":3, "bad":4} # This is a simple vocabulary that maps each unique word to a unique integer index. The <pad> token is used for padding sequences to a fixed length during training.

# Hyperparameters => It means that hyperparameters are the parameters that we set before training the model. They are not learned by the model. They are used to control the training process and the model's performance.
Vocab_size = len(vocab) # This is one of the parameters that we need to set before training the model. It is the size of the vocabulary. It is the number of unique words in the dataset.
EMb_Dim = 8 # This is one of the parameters that we need to set before training the model. It is the dimensionality of the embedding vectors. Dimensionality means the number of features in the embedding vector. For example, if we set the embedding dimension to 8, then each word will be represented by an 8-dimensional vector. The choice of embedding dimension is a hyperparameter that can affect the performance of the model. A larger embedding dimension can capture more information about the words, but it can also lead to overfitting if the dataset is small. A smaller embedding dimension can reduce the risk of overfitting, but it may not capture enough information about the words. Therefore, it is important to choose an appropriate embedding dimension based on the size of the dataset and the complexity of the task.
Hidden_Dim = 16 # This is one of the parameters that we need to set before training the model. It is the dimensionality of the hidden states in the neural network.


# Model 
class Sentimental(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Vocab_size, EMb_Dim) # This is the embedding layer. It takes the vocab size and the embedding dimension as input and creates a look up table of shape (vocab_size, embedding_dim). The look up table is initialized with random values. The embedding layer will learn to update these values during training to capture the semantic meaning of the words.
        self.fc1 = nn.Linear(EMb_Dim, Hidden_Dim) # This is the first fully connected layer. It takes the embedding dimension as input and outputs the hidden dimension. The fully connected layer will learn to update its weights during training to capture the relationships between the words and the sentiment of the sentence.
        self.fc2 = nn.Linear(Hidden_Dim, 2) # This is the final fully connected layer. It takes the hidden dimension as input and outputs the number of classes in the classification task. In this case, we have 2 classes (positive and negative sentiment).
        
    def forward(self, x):
        x = self.embedding(x) # This is the forward pass through the embedding layer. It takes the input tensor of shape (batch_size, sequence_length) and outputs a tensor of shape (batch_size, sequence_length, embedding_dim). Each word in the input tensor is replaced by its corresponding embedding vector from the look up table.
        x = torch.mean(x, dim=1) # This is a simple way to aggregate the word embeddings into a single vector for the entire sentence. We take the mean of the embeddings along the sequence length dimension. This results in a tensor of shape (batch_size, embedding_dim).
       # x = self.fc1(x) # This is the forward pass through the first fully connected layer. It takes the aggregated embedding vector as input and outputs a hidden representation of shape (batch_size, hidden_dim).
        x = torch.relu(x) # This is an activation function that introduces non-linearity into the model. It applies the ReLU function element-wise to the hidden representation.
        # x = self.fc2(x) # This is the forward pass through the final fully connected layer. It takes the hidden representation as input and outputs a tensor of shape (batch_size, num_classes). Each element in this tensor represents the logit for each class.
        return self.fc2(x) # This is the final output of the model. It returns the logits for each class. During training, we will apply a softmax function to these logits to get the probabilities for each class, and then use a loss function to compute the loss and update the model's parameters.
    
model = Sentimental()

sentence = [
    [1,2],
    [3,4],
]

