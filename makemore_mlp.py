import torch
import torch.nn.functional as F

words = open("names.txt" , "r").read().splitlines()

chars = sorted(list(set("".join(words))))

stoi = {s:i+1 for i ,s in enumerate(chars)} ; stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 3 #context length , how many previous chars to take to predict the next one? 
X , Y = [] ,[] #X for previous context , Y for next ch in sequence

for w in words:
    context = [0] * block_size #Create context window with block size length
    """
    print(w)
    """
    for ch in w + ".":
        ix = stoi[ch] #Look up for the index of ch
        X.append(context)
        Y.append(ix)
        """
        print("".join(itos[i] for i in context) , "--->" , itos[ix])
        """
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
nums = Y.nelement()

g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((27,2), generator = g)


"""
print(C[5])
print(F.one_hot(torch.tensor(5), num_classes = 27).float() @ C

print(X[1])

print(C[X][1]) # X = [0,0,5] -> [C[0] , C[0] , C[5] ]
print(C[0])
"""

emb = C[X] # Embedding of X : C[0] is now embeddings of X[0]
emb.shape #Size[X, C.size[1]] : Every index is now embedded with C_dim.

#Weight and bias
W1 = torch.randn((6,100) , generator = g)
b1 = torch.randn((100) , generator = g)

"""
print((emb[:,0,:][0] == C[X][0][0]).all().item()) #True
"""

#Concatenate the context blocks , side by side (dim = 1)
# Not useful for large block_size
"""
cat_ = torch.cat((emb[:,0,:], emb[:,1,:] , emb[:,2,:]), 1)
print(cat_.shape)
"""

#Unbind : Remove a dimension of the tensor, this case dim = 1
#Creates new memory; slower and inefficient than view()
"""
print(torch.cat(torch.unbind(emb, 1), 1).shape)
"""


#a.view = reshape the tensor , will fail if tensor is non contiguous , i.e transposed etc.
#a.reshape is more robust
#a.storage is the way computer stores the elements
"""
a = torch.arange(18)
print(a.view(2,3,3))
print(a.storage)
"""

#True
"""
print((emb.view(32,6) == torch.cat(torch.unbind(emb, 1), 1)).all())
"""

h = torch.tanh(emb.view(-1,6) @ W1 + b1)

W2 = torch.randn((100,27),generator = g)
b2 = torch.randn((27), generator = g)

logits = h @ W2 + b2
counts = logits.exp()
prob = counts / counts.sum(dim = 1 , keepdim = True)
loss = -prob[torch.arange(nums), Y].log().mean()


