import torch
import torch.nn.functional as F
import random

words = open("names.txt" , "r").read().splitlines()

chars = sorted(list(set("".join(words))))

stoi = {s:i+1 for i ,s in enumerate(chars)} ; stoi["."] = 0
itos = {i:s for s,i in stoi.items()}
"""
block_size = 3 #context length , how many previous chars to take to predict the next one? 
X , Y = [] ,[] #X for previous context , Y for next ch in sequence
for w in words:
    context = [0] * block_size #Create context window with block size length
   
    print(w)

    for ch in w + ".":
        ix = stoi[ch] #Look up for the index of ch
        X.append(context)
        Y.append(ix)
  
        print("".join(itos[i] for i in context) , "--->" , itos[ix])

        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
"""

def build_dataset(words):
    block_size = 3 
    X , Y = [] ,[]
    for w in words:
        context = [0] * block_size 

        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

random.seed(42)
random.shuffle(words)

n1= int(0.6 * len(words))
n2 = int(0.8 * len(words))
n3 = len(words)

Xtr, Ytr = build_dataset(words[:n1])
Xdev , Ydev = build_dataset(words[n1:n2])
Xtest , Ytest = build_dataset(words[n2:n3])


g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((27,2), generator = g)


emb = C[Xtest] # Embedding of X : C[0] is now the embeddings of X[0]
emb.shape #Size[X, C.size[1]] : Every index is now embedded with C_dim.

#Weight and bias
W1 = torch.randn((6,100) , generator = g)
b1 = torch.randn((100) , generator = g)

h = torch.tanh(emb.view(-1,6) @ W1 + b1)

W2 = torch.randn((100,27),generator = g)
b2 = torch.randn((27), generator = g)

logits = h @ W2 + b2

"""
counts = logits.exp()
prob = counts / counts.sum(dim = 1 , keepdim = True)
loss = -prob[torch.arange(nums), Y].log().mean()
"""


"""
parameters = [C,W1,b1,W2,b2]
print(sum([p.nelement() for p in parameters]))

"""

#More efficient than manually calculating
#Calculates the max of logits and subtracts it from them ;
# so max of logits become 0 -> no overflow risk
#Also more efficient forward and backward pass
loss = F.cross_entropy(logits,Ytest) 

parameters = [C,W1,b1,W2,b2]
for p in parameters:
    p.requires_grad = True


#candidate learning rates
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

for _ in range(30000):
    #minibatch construction, each loop with random indices
    ix = torch.randint(0, Xtr.shape[0] ,(32,))

    #Forward pass
    emb = C[Xtr[ix]] # (32,3,2)
    h = torch.tanh(emb.view(-1,6) @ W1 + b1) # (32,100)
    logits = h @ W2 + b2 # (32,27)
    loss = F.cross_entropy(logits,Ytr[ix])

    #Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update
    lr = 0.1
    for p in parameters:
        p.data -= lr * p.grad

    #print(loss)

"""
print(logits.max(1))
"""

emb = C[Xdev]
h = torch.tanh(emb.view(-1,6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits,Ydev)
print(loss)
