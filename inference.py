import torch
import torch.nn.functional as F
import pickle

from model import GPT
from config import device, block_size
from preprocessing import encode, decode

with open("data/itos.pkl","rb") as f:
    itos = pickle.load(f)

with open("data/stoi.pkl","rb") as f:
    stoi = pickle.load(f)


model = GPT()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

@torch.no_grad()

def generate(idx, max_new_tokens):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits = model(idx_cond)

        logits = logits[:,-1,:]

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx,next_token),dim=1)
    return idx

while True:

    prompt = input("Enter your prompt to get a response in shakespearean style: ")

    if prompt.lower() ==  "exit":
        break

    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    output = generate(context, max_new_tokens=100)

    generated_text = decode(output[0].tolist())

    print("Generated text:", generated_text)