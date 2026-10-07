import torch
import torch.nn.functional as F
import random

words = open("names.txt" , "r").read().splitlines()

chars = sorted(list(set("".join(words))))

stoi = {s:i+1 for i ,s in enumerate(chars)} ; stoi["."] = 0
itos = {i:s for s,i in stoi.items()}


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

n1= int(0.8 * len(words))
n2 = int(0.9 * len(words))
n3 = len(words)

Xtr, Ytr = build_dataset(words[:n1])
Xdev , Ydev = build_dataset(words[n1:n2])
Xtest , Ytest = build_dataset(words[n2:n3])
X ,Y = build_dataset(words)


g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((27,10), generator = g)


emb = C[Xtest] # Embedding of X : C[0] is now the embeddings of X[0]
emb.shape #Size[X, C.size[1]] : Every index is now embedded with C_dim.

#Weight and bias
W1 = torch.randn((30,200) , generator = g) * (5/3) / (30**0.5)
b1 = torch.randn((200) , generator = g) * 0.01

h = torch.tanh(emb.view(-1,30) @ W1 + b1)

W2 = torch.randn((200,27),generator = g) * 0.01
b2 = torch.zeros((27))

logits = h @ W2 + b2


#More efficient than manually calculating
#Calculates the max of logits and subtracts it from them ;
# So max of logits become 0 -> no overflow risk
#Also more efficient forward and backward pass
loss = F.cross_entropy(logits,Ytest) 

parameters = [C,W1,b1,W2,b2]
for p in parameters:
    p.requires_grad = True


for i in range(200000):
    #minibatch construction, each loop with random indices
    ix = torch.randint(0, Xtr.shape[0] ,(32,))

    #Forward pass
    emb = C[Xtr[ix]] # (32,3,2)
    h = torch.tanh(emb.view(-1,30) @ W1 + b1) # (32,100)
    logits = h @ W2 + b2 # (32,27)
    loss = F.cross_entropy(logits,Ytr[ix])

    #Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data -= lr * p.grad

#Training loss
emb = C[X] #C[torch.cat((Xtr, Xdev,Xtest), dim = 0)]
h = torch.tanh(emb.view(-1,30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits,Y,) # torch.cat((Ytr, Ydev,Ytest), dim = 0
print(loss)


#Cross validation set loss
emb = C[Xdev]
h = torch.tanh(emb.view(-1,30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits,Ydev)
print(loss)

#Sampling
for _ in range(20):
    context_ = [0] * 3
    out = []
    while True:
        emb_ = C[context_]
        h = torch.tanh(emb_.view(1,-1) @ W1 + b1)
        logits = h @ W2 + b2
        counts = logits.exp()
        p = F.softmax(logits)
        ix = torch.multinomial(p , num_samples = 1 , replacement = True, generator = g).item()
        if ix == 0:
            break
        out.append(itos[ix])
        context_ = context_[1:] + [ix]
    print("".join(out))