import torch


# data
batch_size = 32
block_size = 8
train_split = 0.9


# model
n_embd = 128
n_head = 4
n_layer = 4
vocab_size = 65

#training
learning_rate = 1e-3
max_iters = 5000
eval_intervals = 500


device = "cuda" if torch.cuda.is_available() else "cpu"
