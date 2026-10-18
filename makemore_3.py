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
n_hidden = 100
vocab_size = 27

g = torch.Generator().manual_seed(2147483647)

class Linear:

    def __init__(self,fan_in,fan_out, bias = True):
        self.weight = torch.randn((fan_in, fan_out) , generator = g) / (fan_in**0.5)
        self.bias = torch.randn((fan_out,)) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:

    def __init__(self,dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        #parameters , trained with backprop
        self.gamma = torch.ones((dim,))
        self.beta = torch.zeros((dim,))
        
        #buffers, trained with "momentum update"
        self.running_mean = torch.zeros((dim,))
        self.running_var = torch.ones((dim,))

    def __call__(self,x):
        #calculate the forward pass
        if self.training:
            xmean = x.mean(dim = 0, keepdim = True)
            xvar = x.std(dim = 0, keepdim = True)
        else:
            xmean = self.running_mean
            xvar = self.running_var  
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        #update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = self.running_mean * (1-self.momentum) + self.momentum * xmean
                self.running_std = self.running_var * (1-self.momentum) + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    
    def __call__(self,x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

#To make the logits nearly equal initially, set weight and bias to a reasonable minimum
#Do not set weights to 0 because of vanishing gradients
#Kaiming = Initialize W1 as std = gain / (fan_in**0.5) 
#For tanh gain = 5/3 , for ReLu sqrt(2)
fan_in = n_embd * block_size


g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((vocab_size,n_embd), generator = g)

layers = [Linear(fan_in, n_hidden), BatchNorm1d(n_hidden),  Tanh(),
          Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(), 
          Linear(n_hidden, n_hidden),BatchNorm1d(n_hidden),  Tanh(), 
          Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(), 
          Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(), 
          Linear(n_hidden, vocab_size) ,BatchNorm1d(vocab_size)]

with torch.no_grad():
    layers[-1].gamma *= 0.1
    for layer in layers[:-1]:
        if isinstance (layer,Linear):
            layer.weight *= 5/3 #gain

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum([p.nelement() for p in parameters])) #total number of parameters

for p in parameters:
    p.requires_grad = True

batch_size = 32
max_steps = 200000

for i in range(max_steps):
    #minibatch construction, each loop with random indices
    ix = torch.randint(0, Xtr.shape[0] ,(batch_size,) ,generator = g)
    Xb , Yb = Xtr[ix], Ytr[ix]
    #embedding
    emb = C[Xb]
    x = emb.view(emb.shape[0], -1)

    #Forward pass
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x,Yb) #loss function

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters: 
        p.data += -lr*p.grad

    if i%10000 == 0:
        print(f"{i:7d}/{max_steps:7d}/{loss.item():.4f}") 
