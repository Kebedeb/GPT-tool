import wandb

dataset = 'tool'
out_dir = 'out-addition-cot'
eval_interval = 250
eval_iters = 20
log_interval = 1

## Model for GPU training
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256
batch_size = 16
gradient_accumulation_steps = 1
max_iters = 2000
learning_rate = 1e-3
device = 'cuda'
compile = False


hyperparameters = {
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 256,
    "block_size": 256,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "max_iters": 2000,
}
wandb_log = True
wandb_proj = 'addition-tool-use'
wandb_run_name = 'cot-tool'
if wandb_log:
    wandb.init(project=wandb_proj,name= wandb_run_name, config=hyperparameters)
    


