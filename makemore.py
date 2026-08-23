import torch
import torch.nn as nn
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

"""
p = N[0].float()
p = p / p.sum() # probability distribution

#initialize a generator with manual seed -> reproduce the same p every time
g = torch.Generator().manual_seed(2147483647)

p = torch.rand(3, generator = g)
p = p / p.sum()

# sample according to p and generator g -> outputs indices
ix = torch.multinomial(p , num_samples = 1 , replacement = True, generator = g).item()
"""

g = torch.Generator().manual_seed(2147483647)
out = []
for i in range(10):
    ix = 0
    word = ""
    while True:
        p = N[ix].float()
        p = p/p.sum()
        ix = torch.multinomial(p,num_samples = 1 , replacement = True, generator = g).item()
        ch = itos[ix] 
        if ch == ".":
            break
        word += ch
    out.append(word)    

print(out)


