import torch
import torch.nn as nn 
from torch.optim import Adam

from config import (
    learning_rate, 
    max_iters,
    eval_intervals,
    device
)

from model import GPT
from data_loader import get_batch

print("Training on device:", device)

model = GPT()
model = model.to(device)

optimizer = Adam(model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def eval():
    model.eval()
    losses = {}

    for split in ["train", "test"]:
        total_loss = 0.0

        for _ in range(100):
            x,y = get_batch(split)
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            B,S,E = logits.shape

            loss = loss_fn(
                logits.view(B*S,E), y.view(B*S)
                )

            total_loss += loss.item()

        losses[split] = total_loss/100

    model.train()
    return losses


model.train()

for iteration in range(max_iters):

    if iteration % eval_intervals == 0:
        losses = eval()
        print(f"Iteration {iteration}: Train Loss: {losses['train']:.4f}, Test Loss: {losses['test']:.4f}")

    xb, yb = get_batch("train")
    xb = xb.to(device)
    yb = yb.to(device)

    logits = model(xb)
    B,S,E = logits.shape
    loss = loss_fn(
        logits.view(B*S,E), yb.view(B*S) 
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pt")


