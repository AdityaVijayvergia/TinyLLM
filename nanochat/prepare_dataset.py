"""
create a script that is run once (on the CPU machine). It will:

1. Download the datasets.
2. Load the custom tokenizer.
3. Tokenize everything into a massive list of integers.
4. Save it as a binary file with sharding.

This binary file will be uploaded to the A10 GPU.
"""

import os
import pickle
import numpy as np
# import torch
import tiktoken
from datasets import load_dataset
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.common import autodetect_device_type


# device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# device_type = autodetect_device_type() if device_type == "" else device_type
# device = torch.device(device_type)
# print(f"Device type: {device_type}")

# tokenizer = get_tokenizer()

# tokens = tokenizer.encode("hello world") 
# print(tokens)


# Load the tokenizer
with open(".cache/tokenizer/64K/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

tokens = tokenizer.encode("hello world") 
print(tokens)
