import torch
import torch.nn as nn

# Find all the names from the list
words = open("names.txt" , "r").read().splitlines()


N = torch.zeros((27,27) , dtype = torch.int)
# Index all chars -> Char : Index {a:0 , b:1 ...}
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i , s in enumerate(chars)} ;stoi["."]= 0
itos = {i+1:s for i , s in enumerate(chars)} ; itos[0] = "."

#Create the tensor to list the count of each pair
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1 , ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1


P = N.double()
P = P / P.sum(dim = -1, keepdim = True) # (27,27) / (27,1) -> Broadcast : Shape(27,27)



g = torch.Generator().manual_seed(2147483647)
out = []
for i in range(10):
    ix = 0
    word = ""
    while True:
        """
        p = N[ix].float()
        p = p/p.sum()  
        """   
        p = P[ix] 
        ix = torch.multinomial(p,num_samples = 1 , replacement = True, generator = g).item()
        ch = itos[ix] 
        if ch == ".":
            break
        word += ch
    out.append(word)    



