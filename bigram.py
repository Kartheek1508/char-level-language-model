import torch

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

print(N[0])

P = N.float()
P = P/P.sum(1,keepdim = True)
print(P[0])

g = torch.Generator().manual_seed(2147483647)

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
    
