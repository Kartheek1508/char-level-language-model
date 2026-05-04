import torch
import torch.nn.functional as F
import random 

words = open('names.txt','r').read().splitlines()

block_size = 3

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

def build_dataset(words):
    X,Y = [],[]
    for w in words:
        context = [0]*block_size
        for ch in w+'.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X,Y

random.shuffle(words)

i1 = int(len(words)*0.8)
i2= int(len(words)*0.9)


X_train,Y_train = build_dataset(words[:i1])
X_val,Y_val = build_dataset(words[i1:i2])

X_test,Y_test = build_dataset(words[i2:])

g = torch.Generator()
C = torch.randn((27,10)) # acts as the look up table 
W1 = torch.randn((30,200))
b1 = torch.randn(200)
W2= torch.randn(200,27)
b2 = torch.randn(27)

param =[C,W1,W2,b1,b2]


for p in param:
    p.requires_grad = True

for i in range(500_000):

    ix = torch.randint(0,X_train.shape[0],(32,)) #minibatch
    embed = C[X_train[ix]]
    h = torch.tanh(embed.view(-1,30)@W1 +b1)
    logits = h@W2 +b2
    loss = F.cross_entropy(logits,Y_train[ix])

    for p in param:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < 250000 else 0.01
    for p in param:
        p.data -= p.grad*lr

emb = C[X_train]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y_train)
print(f"Training loss : {loss.item()}")

emb = C[X_val]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
val_loss = F.cross_entropy(logits, Y_val)
print(f"Validation loss  = {val_loss}")

g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size 
    while True:
      emb = C[torch.tensor([context])]
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))


emb = C[X_test]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
test_loss = F.cross_entropy(logits, Y_test)
print(f"Test loss  = {test_loss}")