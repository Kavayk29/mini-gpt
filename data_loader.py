import pickle
import torch
from config import batch_size, block_size

train_data = torch.load("data/train_data.pt")
test_data = torch.load("data/test_data.pt")


def get_batch(split):
    data = train_data if split == "train" else test_data
    i = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[k:k + block_size] for k in i])
    y = torch.stack([data[k + 1:k + 1 + block_size] for k in i])
    return x, y