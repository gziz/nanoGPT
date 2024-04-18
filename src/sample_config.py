import random
import time
import torch


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return '\n'.join(f"{k} = {v}" for k, v in self.__dict__.items())

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = int(time.time())
device = 'cuda' if torch.cuda.is_available() else "cpu"
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('src/configurator.py').read())  # overrides from command line or configurator.py.

config = Config(
    init_from=init_from,
    out_dir=out_dir,
    start=start,
    num_samples=num_samples,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_k=top_k,
    seed=seed,
    device=device,
    dtype=dtype,
    compile=compile
)
