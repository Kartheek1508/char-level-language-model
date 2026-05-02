import torch
import torch.nn.functional as F
words = open("names.txt","r").read().splitlines()
chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

N = torch.zeros((27,27),dtype = torch.int32)

for w in words:
    ch = ["."] + list(w) + ['.']
    for ch1,ch2 in zip(ch,ch[1:]):
        i1 = stoi[ch1]
        i2 = stoi[ch2]
        N[i1,i2] += 1



#Bigram model - 
P = (N+1).float()
P = P/P.sum(1,keepdim = True)


g = torch.Generator().manual_seed(2147483647)

print("10 predictions using bigram model")

for i in range(10):
    s = []
    i = 0
    while True:

        p = P[i]
        i = torch.multinomial(p,num_samples = 1, replacement = True,generator = g).item()
        s.append(itos[i])
        if i == 0:
            break
    print(''.join(s))

log_likelihood = 0.0
n = 0

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1

nll = -log_likelihood
bigram_loss = nll/n

#Neural Net - 
    
x,y = [],[]
for w in words:
    ch = ["."] + list(w) + ['.']
    for ch1,ch2 in zip(ch,ch[1:]):
        i1 = stoi[ch1]
        i2 = stoi[ch2]
        x.append(i1)
        y.append(i2)

x = torch.tensor(x)
y = torch.tensor(y)
num = x.nelement()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27),generator=g,requires_grad=True)

for i in range(1000):
    x_enc = F.one_hot(x,num_classes=27).float()
    logits = x_enc @ W
    counts = logits.exp()
    prob = counts/counts.sum(1,keepdim=True)
    loss = -prob[torch.arange(num),y].log().mean()+ 0.01*(W**2).mean().item()
    if i%100 ==0:
        print(loss)

    W.grad = None
    loss.backward()

    W.data += -50 * W.grad


g = torch.Generator().manual_seed(2147483647)

print("10 predictions using Nueral net: ")

for i in range(10):
  
  out = []
  ix = 0
  while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W 
    counts = logits.exp()
    p = counts / counts.sum(1, keepdims=True)
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))


print(f"Neural network's negative log likelihood - {loss}")

print(f"Bigram negative log likelihood - {bigram_loss}")
