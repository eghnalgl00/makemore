import torch
import torch.nn.functional as F
import random

words = open("names.txt" , "r").read().splitlines()

chars = sorted(list(set("".join(words))))

stoi = {s:i+1 for i ,s in enumerate(chars)} ; stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 3 

def build_dataset(words):
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

n_embd = 10
n_hidden = 200
vocab_size = 27

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((vocab_size,n_embd), generator = g)

#To make the logits nearly equal initially, set weight and bias to a reasonable minimum
#Do not set weights to 0 because of vanishing gradients
W1 = torch.randn((n_embd * block_size,n_hidden) , generator = g)
b1 = torch.randn((n_hidden) , generator = g)
W2 = torch.randn((n_hidden,vocab_size),generator = g) * 0.01
b2 = torch.zeros((vocab_size)) * 0

parameters = [C,W1,b1,W2,b2]
for p in parameters:
    p.requires_grad = True

batch_size = 32
max_steps = 200000

for i in range(max_steps):
    #minibatch construction, each loop with random indices
    ix = torch.randint(0, Xtr.shape[0] ,(batch_size,) ,generator = g)

    Xb , Yb = Xtr[ix], Ytr[ix]

    #Forward pass
    emb = C[Xb]
    embcat = emb.view(emb.shape[0] , -1)
    hpreact = embcat @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits,Yb)

    #Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data -= lr * p.grad

    if i % 10000 == 0:
        print(f"{loss.item():.4f}")

with torch.no_grad():
    def split_loss(split):
        X , Y = {"train":(Xtr,Ytr) , "dev":(Xdev, Ydev), "test":(Xtest, Ytest)}[split.lower()]
        emb = C[X]
        embcat = emb.view(emb.shape[0] , -1)
        hpreact = embcat @ W1 + b1
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits,Y)
        print(split , loss.item())
split_loss("train")
split_loss("dev")

g = torch.Generator().manual_seed(2147483647 + 10)

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