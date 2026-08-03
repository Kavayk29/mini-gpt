import torch
import urllib.request
import pickle

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url,"data/input.txt")


with open("data/input.txt","r") as f:
    text = f.read()

list1 = sorted(set(text))

stoi = {ch:i for i,ch in enumerate(list1)}
itos = {i:ch for i,ch in enumerate(list1)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

data = encode(text)
data = torch.tensor(data,dtype=torch.long)


n = len(data) * 0.9

train_data = data[:int(n)]

test_data = data[int(n):]

torch.save(train_data,"data/train_data.pt")
torch.save(test_data,"data/test_data.pt")

with open("data/stoi.pkl","wb") as f:
    pickle.dump(stoi,f)

with open("data/itos.pkl","wb") as f:
    pickle.dump(itos,f)