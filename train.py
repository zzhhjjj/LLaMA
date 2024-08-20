import sys
import time
from tools.debug.debug_llama import DebugMyCausalSelfAttentionforward, DebugMyDecoderLayerforward, DebugMyLLaMAforward
from src.model.llama3 import LLaMA
from tools.debug.replace_attn import DebugDecoderLayerforward, DebugRoPEforward, DebugSDPAforward
import torch 
from dataclasses import dataclass
from src.weights.weights import load_weights_to_dict, copy_weights_to_model
from transformers import AutoTokenizer
import os
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from src.data.data import tokenize_dataset, get_dataloader


# to match the output of transformers model, set MERGED_QKV_WEIGHT to 0 is necessary
os.environ['DATA_TYPE'] = 'bfloat16' # bfloat16/float32
os.environ['MERGED_QKV_WEIGHT'] = '0' # 1/0
os.environ['MERGED_GATE_UP_WEIGHT'] = '1' # 1/0
os.environ['TRITONRMSNORM'] = '1'
os.environ['FLASH_ROPE'] = '1'
os.environ['ATTENTION'] = 'FLASH' # SDPA/FLASH

# set device and dtype
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16 if os.getenv('DATA_TYPE', 'bfloat16') == 'bfloat16' else torch.float32

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Initialize model, loss function, and optimizer
## GPT2 base model
sequence_length = 1024
@dataclass
class LLaMAConfig:
    batch_size: int = 1
    max_position_embeddings: int = sequence_length
    hidden_dim: int = 768
    intermediate_dim: int = 3072
    vocab_size = 50304 #  https://x.com/karpathy/status/1621578354024677377 still true? 
    num_key_values: int = 4
    num_heads: int = 12
    num_layers: int = 12
    rope_theta: float = 500000.0
    torch_dtype: str = 'bfloat16'
    rms_norm_eps: float = 1e-5

dataset = load_dataset("roneneldan/TinyStories", split='train')
tokenized_dataset = tokenize_dataset(dataset, tokenizer, text_column_name = 'text', sequence_length = sequence_length, dataset_processing_num_proc_per_process = 64)

# DataLoader
batch_size = 8
shuffle = False
dataloader = get_dataloader(tokenized_dataset, batch_size, shuffle)

config = LLaMAConfig()
model = LLaMA(config).to(dtype).to(device)
criterion = nn.CrossEntropyLoss()  # or any other suitable loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
step = 0
num_epochs = 5
total_tokens_processed = 0
start_time = time.time()  # Start time measurement

for epoch in range(num_epochs):
    for data in dataloader:
        input_ids, label_ids = data['input_ids'].to(device) , data['label_ids'].to(device)
        
        # Forward pass
        outputs = model(input_ids)
        
        # adjust shape for the loss
        B, T, C = outputs.size()
        outputs = outputs.view(B*T, C)
        label_ids = label_ids.view(B*T)
        
        # Compute the loss
        loss = criterion(outputs, label_ids)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Increment step counter
        step += 1
        
        # Print the loss for each step
        tokens_processed = B * T  # B is batch size, T is sequence length
        total_tokens_processed += tokens_processed
        
        # Print the loss for each step
        if step % 10 == 0:
            print(f'Step [{step}], Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Tokens/s : {total_tokens_processed / (time.time() - start_time):.2f}')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")





