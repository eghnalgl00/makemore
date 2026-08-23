import torch
import matplotlib.pyplot as plt

# Find all the names from the list
words = open("names.txt" , "r").read().splitlines()

"""
b = {}
#iterate throgh pairs in words , with Start and End .
for w in words:
    chs = ["<S>"] + list(w) + ["<E>"]
    for ch1 , ch2 in zip(chs, chs[1:]):
        bigram = (ch1 ,ch2)
        b[bigram] = 1 + b.get(bigram,0)
        
sorted_b = dict(sorted(b.items(), key=lambda kv: -kv[1])
"""

N = torch.zeros((27,27) , dtype = torch.int)
# Index all chars -> Char : Index {a:0 , b:1 ...}
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i , s in enumerate(chars)}
stoi["."]= 0
itos = {i+1:s for i , s in enumerate(chars)} ; itos[0] = "."

#Create the tensor to list the count of each pair
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1 , ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1