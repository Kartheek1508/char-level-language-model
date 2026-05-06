import torch
import torch.nn.functional as F
import random 

words = open('names.txt','r').read().splitlines()

block_size = 3

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

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


n_embd = 10 
n_hidden = 200

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0
bn_gain = torch.ones(1,n_hidden)
bn_bias = torch.zeros(1,n_hidden)
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))


param =[C,W1,W2,b1,b2,bn_gain,bn_bias]


for p in param:
    p.requires_grad = True

max_steps = 300000
batch_size = 32

#Training
for i in range(max_steps):

    ix = torch.randint(0,X_train.shape[0],(batch_size,)) #minibatch
    embed = C[X_train[ix]]
    h_pre_act = embed.view(embed.shape[0],-1)@W1 +b1
    bnmeani = h_pre_act.mean(0,keepdim=True)
    bnstdi = h_pre_act.std(0,keepdim=True)
    h_pre_act = bn_gain*(h_pre_act-bnmeani)/bnstdi + bn_bias
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    h = torch.tanh(h_pre_act)
    logits = h@W2 +b2
    loss = F.cross_entropy(logits,Y_train[ix])

    for p in param:
        p.grad = None
    loss.backward()
    lr = 0.1 if i < 150000 else 0.01
    for p in param:
        p.data -= p.grad*lr

    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')

def split_loss(split):
  x,y = {
    'train': (X_train, Y_train),
    'val': (X_val, Y_val),
    'test': (X_test, Y_test),
  }[split]
  emb = C[x]
  embcat = emb.view(emb.shape[0], -1) 
  hpreact = embcat @ W1  #+ b1

  hpreact = bn_gain * (hpreact - bnmean_running) / bnstd_running + bn_bias
  h = torch.tanh(hpreact) 
  logits = h @ W2 + b2 
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss("val")
split_loss('test')

g = torch.Generator().manual_seed(2147483647)

#generating 
for _ in range(20):
    
    out = []
    context = [0] * block_size 
    while True:
        emb = C[torch.tensor([context])]
        hpreact = emb.view(emb.shape[0], -1) @ W1 + b1
        hpreact = bn_gain * (hpreact - bnmean_running) / bnstd_running + bn_bias
        h = torch.tanh(hpreact) 

        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    
    print(''.join(itos[i] for i in out))
