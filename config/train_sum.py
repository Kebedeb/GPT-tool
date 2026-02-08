dataset = 'tool'
out_dir = 'out-addition'
eval_interval = 250
eval_iters = 20
log_interval = 1

## Model for GPU training
n_layer = 4
n_head = 4
n_embd = 256
block_size = 128
batch_size = 16
gradient_accumulation_steps = 1
max_iters = 2000
learning_rate = 1e-3
device = 'cuda'
compile = False
