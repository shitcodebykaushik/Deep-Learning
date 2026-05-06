# Implementation the High-Performance Transformenr with the scaled dot product attention 
# `Scaled Dot-Product-Attention => It is  a Pytorch function that computes the attention - the core operation in transformenrs models,Instead of writing the attention math yourself step by step,you call this one function and pytorch handles it efficienctly under the hood .
# Attention is (QK^T)/sqrt(d_k) where Q is the query matrix, K is the key matrix, and d_k is the dimension of the key vectors. The output is a weighted sum of the value vectors, where the weights are determined by the similarity between the query and key vectors.
# Attention is the mechanism that lest a model decide how much to focus on each word when processing a sentence .
# Before attention, models read text left to right, word by word (RNNs). They would often forget early words by the time they reached the end of a long sentence.
# Attention solves this — every word can directly look at every other word in one step, no matter how far apart they are. That's why transformers are so powerful.
# The scaled dot function is a high performance pytorch utility that combines several setps - matrix multiplication,scaling ,softmax and dropout into a single optized cell 


import torch 
import torch.nn.functional as F

B,H,L,E = 2,8,128,64

# Batch = THis means we  are processing 2 sentences/ sequence into the GPU simultaneously .
# Heads = The number of the attention heads.This allows the model to focus on 8 different things at once . And head means in the simple words the number of different attention patterns the model can learn to look at the input data.
# Length = This means sequence length or the number of words in each sentence. Here we have 128 words in each sentence.
# Embedding = This means the dimension size . Each single token is represented by a list of 64 number that describe its meaning .
 # When we combine all these three BHLE together then create the data structure with the total of 1638400 numbers(2*8*128*64) that represent the input data for the attention mechanism to process. 
 


# In ai world every input token is converted into three different version of itself: A query, a key, and a value. These are like different perspectives of the same word that the model uses to figure out how much attention to pay to it.
# So in general we are creating the  three identical rooms full of random  full of random data and the attenction function will later try to find the connection between them .

query = torch.rand(B,H,L,E)
key = torch.rand(B,H,L,E)
value = torch.rand(B,H,L,E)

print('\n')
output = F.scaled_dot_product_attention(query,key,value)   # This is the basic attenction 
print(output.shape)
print("\n")
output = F.scaled_dot_product_attention(query, key, value, is_causal=True) # This  is like  everyone can talk,but you are forbidden from looking at the pages ahead of where you are currently reading . You can only look at what you have already read . This is how AI learns to predict the next word without cheating  by seeing the future .
print(output)
print("\n")
# This forces the rest of the group to work harder and not to rely on the just  one smart person .
output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.1)
print(output)

