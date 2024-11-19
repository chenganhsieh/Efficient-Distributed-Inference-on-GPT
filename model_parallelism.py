import torch.distributed as dist
import torch
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import time
import psutil
import os
from evaluate import evaluate
from metrics import measure_inference_metrics
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

class GPT2Segment1(nn.Module):
    def __init__(self, original_model):
        super(GPT2Segment1, self).__init__()
        # Register the first 12 transformer blocks
        self.transformer_blocks = nn.ModuleList(original_model.transformer.h[:12])
        # Recreate embeddings to ensure they're properly registered
        self.wte = nn.Embedding.from_pretrained(original_model.transformer.wte.weight.clone())
        self.wpe = nn.Embedding.from_pretrained(original_model.transformer.wpe.weight.clone())
        # Copy other necessary components
        self.dropout = original_model.transformer.drop
        self.config = original_model.config

    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device
        # Create position ids
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # Get embeddings
        inputs_embeds = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.dropout(inputs_embeds)
        # **Modify attention_mask**
        if attention_mask is not None:
            # Convert attention_mask to float and invert it
            attention_mask = attention_mask.to(device).unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # Convert to the same dtype as hidden_states
            attention_mask = (1.0 - attention_mask) * -10000.0  # Apply scaling
        else:
            attention_mask = None
        # Apply transformer blocks
        for block in self.transformer_blocks:
            outputs = block(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]
        return hidden_states


class GPT2Segment2(nn.Module):
    def __init__(self, original_model):
        super(GPT2Segment2, self).__init__()
        # Register the remaining transformer blocks
        self.transformer_blocks = nn.ModuleList(original_model.transformer.h[12:])
        # Copy layer normalization and output head
        self.ln_f = nn.LayerNorm(original_model.transformer.ln_f.normalized_shape, eps=original_model.transformer.ln_f.eps)
        self.lm_head = nn.Linear(original_model.lm_head.in_features, original_model.lm_head.out_features, bias=False)
        self.lm_head.weight = original_model.lm_head.weight
        self.dropout = original_model.transformer.drop
        self.config = original_model.config

    def forward(self, hidden_states, attention_mask=None):
        device = next(self.parameters()).device
        # **Modify attention_mask**
        if attention_mask is not None:
            # Convert attention_mask to float and invert it
            attention_mask = attention_mask.to(device).unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # Convert to the same dtype as hidden_states
            attention_mask = (1.0 - attention_mask) * -10000.0  # Apply scaling
        else:
            attention_mask = None
            attention_mask = attention_mask.to(device)
        # Apply transformer blocks
        for block in self.transformer_blocks:
            outputs = block(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]
        # Apply layer normalization
        hidden_states = self.ln_f(hidden_states)
        # Get logits
        logits = self.lm_head(hidden_states)
        return logits

# Load the Penn Treebank dataset
dataset = load_dataset('ptb_text_only', 'penn_treebank')
valid_texts = dataset['validation']['sentence']

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples, return_tensors='pt', padding=True, truncation=True, max_length=128)

valid_encodings = tokenize_function(valid_texts)

# Create DataLoader
valid_dataset = torch.utils.data.TensorDataset(valid_encodings['input_ids'], valid_encodings['attention_mask'])
valid_loader = DataLoader(valid_dataset, batch_size=8)


# Assume two GPUs are available
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')

# Load the original model
original_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# Initialize segments
model_segment1 = GPT2Segment1(original_model).to(device0)
model_segment2 = GPT2Segment2(original_model).to(device1)


def model_parallel_inference(input_ids, attention_mask=None):
    # Move inputs to device0
    input_ids = input_ids.to(device0)
    attention_mask = attention_mask.to(device0) if attention_mask is not None else None

    # Forward pass through segment 1
    hidden_states = model_segment1(input_ids, attention_mask=attention_mask)

    # Move hidden_states to device1
    hidden_states = hidden_states.to(device1)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device1)

    # Forward pass through segment 2
    logits = model_segment2(hidden_states, attention_mask=attention_mask)
    return logits

# Example inference
def inference(model_fn, dataloader):
    model_segment1.eval()
    model_segment2.eval()
    latencies = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            print(f"Batch:{idx}|{len(dataloader)}", end="\r")
            input_ids = batch[0]
            attention_mask = batch[1]

            start_time = time.time()
            logits = model_fn(input_ids, attention_mask)
            latency = time.time() - start_time
            latencies.append(latency)

    avg_latency = sum(latencies) / len(latencies)
    throughput = len(dataloader.dataset) / sum(latencies)
    memory_usage0 = torch.cuda.max_memory_allocated(device0) / (1024 ** 3)
    memory_usage1 = torch.cuda.max_memory_allocated(device1) / (1024 ** 3)
    total_memory_usage = memory_usage0 + memory_usage1
    print(f"Average Latency: {avg_latency * 1000:.2f} ms/query")
    print(f"Throughput: {throughput:.2f} queries/second")
    print(f"Memory Usage: {total_memory_usage:.2f} GB")


inference(model_parallel_inference, valid_loader)