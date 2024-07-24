import torch


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return '\n'.join(f"{k} = {v}" for k, v in self.__dict__.items())

# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'

# data
# batch_size = 12
# block_size = 1024
# gradient_accumulation_steps = 5 * 8
dataset = 'openwebtext'
total_batch_size = 4096
batch_size = 4
block_size = 1024
gradient_accumulation_steps = total_batch_size // (batch_size * block_size)

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# optimizer
learning_rate = 6e-4
max_iters = 600_000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600_000
min_lr = 6e-5

# DDP settings
backend = 'nccl'

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('src/configurator.py').read())  # overrides from command line or configurator.py.



config = Config(
    out_dir=out_dir,
    eval_interval=eval_interval,
    log_interval=log_interval,
    eval_iters=eval_iters,
    eval_only=eval_only,
    always_save_checkpoint=always_save_checkpoint,
    init_from=init_from,
    wandb_log=wandb_log,
    wandb_project=wandb_project,
    wandb_run_name=wandb_run_name,
    dataset=dataset,
    gradient_accumulation_steps=gradient_accumulation_steps,
    batch_size=total_batch_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
    learning_rate=learning_rate,
    max_iters=max_iters,
    weight_decay=weight_decay,
    beta1=beta1,
    beta2=beta2,
    grad_clip=grad_clip,
    decay_lr=decay_lr,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    min_lr=min_lr,
    backend=backend,
    device=device,
    dtype=dtype,
    compile=compile
)

# -----------------------------------------------------------------------------