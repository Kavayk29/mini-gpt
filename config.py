import torch

batch_size = 32
block_size = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

train_split = 0.9