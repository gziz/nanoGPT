"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import sys
import torch
import tiktoken
from model import GPTConfig, GPT
from sample_config import config


def initialize_device():
# Determine device type and precision type based on configuration
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)
    return device_type, dtype, ctx

def load_model():
    # Model initialization
    if config.init_from == 'resume':
        # Initialize from a model saved in a specific directory
        ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model, checkpoint
    elif config.init_from.startswith('gpt2'):
        # Initialize from a given GPT-2 model variant
        return GPT.from_pretrained(config.init_from, dict(dropout=0.0)), None

def setup_encoding_decoding(checkpoint):
    load_meta = False
    if config.init_from == 'resume' and ('config' in checkpoint) and ('dataset' in checkpoint['config']):
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # Default to GPT-2 encodings
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={""})
        decode = lambda l: enc.decode(l)
    return encode, decode

def main():
    # Ensure random seeds and computational settings
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul operations
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cuDNN operations

    
    device, dtype, ctx = initialize_device()
    model, checkpoint = load_model()

    # Set model to evaluation mode and assign to the appropriate device
    model.eval()
    model.to(config.device)
    if config.compile:
        model = torch.compile(model)  # Optional compilation for performance, requires PyTorch 2.0

    # Load metadata if available
    encode, decode = setup_encoding_decoding(checkpoint)

    # Encode the beginning of the prompt
    if config.start.startswith('FILE:'):
        with open(config.start[5:], 'r', encoding='utf-8') as f:
            config.start = f.read()
    start_ids = encode(config.start)
    x = torch.tensor(start_ids, dtype=torch.long, device=config.device)[None, ...]

    # Run the generation loop
    with torch.no_grad():
        with ctx:
            for k in range(config.num_samples):
                print(config.start, sep="")
                for y in  model.generate(x, config.max_new_tokens, config.temperature, config.top_k):
                    print(decode(y[0].tolist()), end="")
                    sys.stdout.flush()

                # print(decode(y[0].tolist()))
                print('\n---------------')

if __name__ == '__main__':
    main()