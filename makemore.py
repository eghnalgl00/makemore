import torch
import torch.nn.functional as F

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

P = (N+1).float() #Model Smoothing. Fake counts to smooth the distribution.


#Generate samples given the probabilities of bigrams
g = torch.Generator().manual_seed(2147483647)


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

logits = xenc@W #log-counts
counts = logits.exp()
probs = counts / counts.sum(dim=-1, keepdim=True) #Softmax


#Find loss
loss = -probs[torch.arange(nums),ys].log().mean()

#Backward pass
W.grad = None  # Set gradient to zero
print(loss)

epochs = 100
lr = 50
for epoch in range(epochs):
    # Forward pass
    logits = xenc@W #log-counts
    counts = logits.exp()
    probs = counts / counts.sum(dim=-1, keepdim=True) #Softmax
    loss = -probs[torch.arange(nums),ys].log().mean() + 0.01 * (W**2).mean() #Regularization
    
    #backward pass
    loss.backward()

    #update
    with torch.no_grad():
        W -= lr * W.grad

    #Zero-grad
    W.grad.zero_()
    if epoch % 10 == 0:
        print(f"{loss.item():.4f}")


#Sample from Neural Network
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0
    
    while True:  
        xenc = F.one_hot(torch.tensor([ix]), num_classes = 27).float()
        logits = xenc@W
        counts = logits.exp()
        p = counts / counts.sum(dim = -1, keepdim = True)

        ix = torch.multinomial(p , num_samples = 1 , replacement = True ,  generator = g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))    