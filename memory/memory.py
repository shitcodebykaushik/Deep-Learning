# The core  desing principle of memory is always the same .How much past information to retain ,how to compress it and hownto selectively retrive what relevant . 
# In a basic neural network, memory is stored in the weights and biases 
# Things are encoded in the weights .
# After training the network remember what it learned through these parameters .
# Skip the RNN network for the memeory 

# Modern  Ai system uses the Transformer architecture which uses the attention and the context windows . They rember the relevant parts of the inpit by focusing attention on importatn words or tokens .
# A neural network memory is not like human or computer memeory instead it is learned weights and the hidden states or attention based context storage .
# Learned Memory (Weights ) This is models long term memory . During the training the network adjusts billion paramets. These weights store the grammer ,facts,reasoning pattern,coding knowledge and the language structure .
# All of this is compressed into numerical weights that the model can use to generate responses .
# This is the model short-term memory . For example in transformer architecture when you say that My dog is MAX  and yhrn you ask what is the name of the my dog ? The model can retrieve the name MAX from the context window and give you the correct answer . This is how the model can maintain a conversation and keep track of the relevant information across multiple turns of dialogue .
# The most important thing is that transformer uses self attention mechanism ,instead of using one word at a time like RNNs ,the model looks at all relevant words  together .
# The model comapres queries and the key to decide which words should pay attention to which other words 
# This created dynamic memory during infernce . 
# The models have the context limits .
# If the information falss outside the context window the model usually forgets it . 
# Token are the basic unit of the transformer model and the exact split depends in the tokenizers .
# The neural network works with the numbers now raw text , so the pipeline is Text->Tokens->IDs->Vectors->Neural Networks .
# Each token is converted into vector callend embedding and these vectors capture meaning and relationship .
# KV cache (Key-Value Cacher) is very important optimization used in the modern Transformer models to make text generation much faster .
# Without KV cache everytime the model generates a new token it would have to recompute attention for the entire conversation again .
# With KV cache the model stores the key and value vectors from previous tokens in the conversation . So when it generates a new token it can reuse those cached vectors instead of recomputing them from scratch . This significantly reduces the computational overhead and allows for much faster response times , especially in long conversations where the context window is large .

import torch 
import torch.nn as nn 

import torch
import torch.nn.functional as F

class SimpleAttention(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Linear layers to create Q, K, and V from the input embeddings
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        # 1. Project inputs to Q, K, V
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 2. Calculate Dot-Product Scores (Similarity)
        # We use transpose(-2, -1) to align the matrix dimensions for multiplication
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model))

        # 3. Apply Softmax to get Attention Weights (Probabilities)
        attention_weights = F.softmax(scores, dim=-1)

        # 4. Multiply weights by V to get the context-aware output
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# Configuration
d_model = 16  # Dimension of our "meaning" vectors
attention_layer = SimpleAttention(d_model)


# Manual Text: "My dog is Max"
# Let's represent 4 tokens with random vectors of size d_model (16)
# In a real model, these would come from an Embedding layer.
input_text_vectors = torch.randn(1, 4, d_model) 

# Run the attention mechanism
output, weights = attention_layer(input_text_vectors)

print("Attention Weights Shape:", weights.shape) 
# Results in [1, 4, 4] -> Each of the 4 words looking at each of the 4 words.

print("Attention focus for 'Max':", weights[0, 3, :])