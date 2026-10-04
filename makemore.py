import torch
import torch.nn.functional as F

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
stoi = {s:i+1 for i , s in enumerate(chars)} ;stoi["."]= 0
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


P = (N+1).float() #Model Smoothing. Fake counts to smooth the distribution.


#Generate samples given the probabilities of bigrams
g = torch.Generator().manual_seed(214748364)
"""
out = []
for i in range(10):
    ix = 0
    word = ""
    while True:
        p = N[ix].float()
        p = p/p.sum()   
        p = P[ix] 
        ix = torch.multinomial(p,num_samples = 1 , replacement = True, generator = g).item()
        ch = itos[ix] 
        if ch == ".":
            break
        word += ch
    out.append(word)

print(out)
"""


#Log Likelihood : log(a*b*c) = log(a) + log(b) + log(c) 
log_likelihood = 0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1 , ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1 , ix2]
        logprob = prob.log()
        log_likelihood += logprob
        n += 1
        """
        print(f"{ch1}{ch2} : {prob:.4f} {logprob:.4f}")
        """


"""
print(f"log_likelihood = {log_likelihood}")

nll = -log_likelihood
avg_nll = nll/n
"""


#create the training set of bigrams (x,y)
xs,ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1 , ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
nums = xs.nelement()

# One-hot encoding for vectorization , the integers themselves would not be useful
xenc = F.one_hot(xs,num_classes = 27).float()
W = torch.randn((27,27) , generator = g , requires_grad = True)
"""
print((xenc@W)[3,13])
print(torch.sum(xenc[3] * W[:,13]))
print(itos[(torch.argmax((torch.exp(xenc@W))[0])).item()])
"""
logits = xenc@W #log-counts
counts = logits.exp()
probs = counts / counts.sum(dim=-1, keepdim=True) #Softmax
#print(itos[torch.argmax(probs[1]).item()])

"""
nll = torch.zeros(1)
for i in range(probs.size()[0]):
    nll = nll  + -(probs[i][ys[i]].log())
nll = nll / probs.size()[0]
print(nll)
"""

print(ys.shape, xs.shape)
#Find loss
loss = -probs[torch.arange(nums),ys].log().mean()

#Backward pass
W.grad = None  # Set gradient to zero
print(loss)

epochs = 1
lr = 50
for epoch in range(epochs):
    # Forward pass
    logits = xenc@W #log-counts
    counts = logits.exp()
    probs = counts / counts.sum(dim=-1, keepdim=True) #Softmax
    loss = -probs[torch.arange(ys.shape[0]),ys].log().mean()
    
    #backward pass
    loss.backward()

    #update
    with torch.no_grad():
        W -= lr * W.grad

    #Zero-grad
    W.grad.zero_()
    if epoch % 10 == 0:
        print(f"{loss.item():.4f}")

