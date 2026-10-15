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
#Kaiming = Initialize W1 as std = gain / (fan_in**0.5) 
#For tanh gain = 5/3 , for ReLu sqrt(2)
fan_in = n_embd * block_size
W1 = torch.randn((fan_in,n_hidden) , generator = g) * 5/3 / (fan_in ** 0.5)

#We add b1 , then calculate mean and then subtract it
#Thus b1 gets added and subtracted all the time
#But it does not have any effect , just unnecessary calculation
#It's grad is zero, since it affects nothing
#bnbias handles the bias
#b1 = torch.randn((n_hidden) , generator = g) * 0.01

W2 = torch.randn((n_hidden,vocab_size),generator = g) * 0.01
b2 = torch.zeros((vocab_size)) * 0

#Batch normalization gain and bias
#Forcing each neuron to be unit Gaussian may inhibit learning
#Thus we let them adjust themselves
#.requires_grad = True
#Each of them are differentiable thus they can learn
bngain = torch.ones((1,n_hidden))
bnbias = torch.ones((1,n_hidden))
bnmean_running = torch.zeros((1,n_hidden))
bnstd_running = torch.ones((1,n_hidden))

#to prevent division by zero in BatchNorm
epsilon = 1e-5

#momentum determines how much a new batch will weigh in running mean and std
#if the batch size is large , they usually have close mean and std values so momentum can be higher
#but if batch size is small enough , a large momentum may inhibit the mean and std from converging to actual one
#PyTorch default = 0.1
momentum = 0.001

parameters = [C,W1,W2,b2 , bngain, bnbias]
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
    hpreact = embcat @ W1 #+ b1
    
    #Batch Normalization Layer

    #hpreact depends on the batch a lot, which is randomly chosen (coupling)
    #Makes it harder for NN to memorize the input
    #Therefore it works like a regularizer
    #We do not like its coupling property , it might damage the learning
    #But regularization effect makes it hard to replace
    #Has some alternatives ; LayerNorm, Group Normalization etc.
    bnmeani = hpreact - hpreact.mean(dim = 0, keepdim = True)
    bnstdi = hpreact.std(dim = 0, keepdim = True)
    hpreact = bngain * (hpreact - bnmean_running) / (bnstd_running + epsilon) + bnbias

    #Update the values of running mean and std
    #We will use it in the inference stage
    #it is not time efficient to estimate these after traning
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + momentum * bnmeani
        bnstd_running = 0.999 * bnstd_running + momentum * bnstdi

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


#if we always use running mean and std , during inference the model would wait for a batch
#Thus after training, we use a learned and static std and mean
#BUT, it is expensive and not time efficient to estimate these after training
"""
with torch.no_grad():
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    bnmean = hpreact.mean(0 , keepdim = True)
    bnstd = hpreact.std(0 , keepdim = True)
    hpreact = (hpreact - bnmean) / bnstd
"""
with torch.no_grad():
    def split_loss(split):
        X , Y = {"train":(Xtr,Ytr) , "dev":(Xdev, Ydev), "test":(Xtest, Ytest)}[split.lower()]
        emb = C[X]
        embcat = emb.view(emb.shape[0] , -1)
        hpreact = embcat @ W1
        #Use running mean and std
        hpreact = bngain*(hpreact - bnmean_running) / bnstd_running + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits,Y)
        print(split , loss.item())

g = torch.Generator().manual_seed(2147483647 + 10)


"""
#Sampling
for _ in range(20):
    context_ = [0] * 3
    out = []
    while True:
        emb_ = C[context_]
        h = torch.tanh(emb_.view(1,-1) @ W1)
        logits = h @ W2 + b2
        counts = logits.exp()
        p = F.softmax(logits , dim = 1)
        ix = torch.multinomial(p , num_samples = 1 , replacement = True, generator = g).item()
        if ix == 0:
            break
        out.append(itos[ix])
        context_ = context_[1:] + [ix]
    #print("".join(out))
"""