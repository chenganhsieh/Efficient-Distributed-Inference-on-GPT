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
        self.dtype = original_model.dtype

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        # Create position ids
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        # Get embeddings
        inputs_embeds = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.dropout(inputs_embeds)
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        
        # **Modify attention_mask**
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Apply transformer blocks
        for block in self.transformer_blocks:
            outputs = block(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]

        return hidden_states, attention_mask


class GPT2Segment2(nn.Module):
    def __init__(self, original_model):
        super(GPT2Segment2, self).__init__()
        # Register the remaining transformer blocks
        self.transformer_blocks = nn.ModuleList(original_model.transformer.h[12:])
        # Copy layer normalization and output head
        self.ln_f = original_model.transformer.ln_f
        self.lm_head = nn.Linear(original_model.lm_head.in_features, original_model.lm_head.out_features, bias=False)
        self.lm_head.weight = original_model.lm_head.weight
        self.config = original_model.config
        self.dtype = original_model.dtype

    def forward(self, hidden_states, attention_mask=None):
        # Apply transformer blocks
        for block in self.transformer_blocks:
            outputs = block(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]
        
        # Apply layer normalization
        hidden_states = self.ln_f(hidden_states)
        # Get logits
        logits = self.lm_head(hidden_states)
        return logits
    
def model_parallel_inference_with_loss(input_ids, attention_mask=None, labels=None):
    # Move inputs to device0
    input_ids = input_ids.to(device0)
    attention_mask = attention_mask.to(device0) if attention_mask is not None else None
    labels = labels.to(device1) if labels is not None else None  # Labels needed on device1 for loss calculation

    # Forward pass through segment 1
    hidden_states, attention_mask = model_segment1(input_ids, attention_mask=attention_mask)
    # Move hidden_states to device1
    hidden_states = hidden_states.to('cpu').to(device1)
    if attention_mask is not None:
        attention_mask = attention_mask.to('cpu').to(device1)

    # Forward pass through segment 2
    logits = model_segment2(hidden_states, attention_mask=attention_mask)

    # Compute loss if labels are provided
    if labels is not None:
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()  # Shift labels to align with logits
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return logits, loss
    else:
        return logits, None

# Load the Penn Treebank dataset
dataset = load_dataset('ptb_text_only', 'penn_treebank')
valid_texts = dataset['validation']['sentence']

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples, return_tensors='pt', padding='max_length', max_length=128)

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


def inference(model_fn, dataloader):
    model_segment1.eval()
    model_segment2.eval()
    latencies = []
    total_loss = 0
    total_tokens = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            print(f"Batch:{idx+1}/{len(dataloader)}", end="\r")
            input_ids = batch[0]
            attention_mask = batch[1]

            # Prepare labels
            labels = input_ids.clone()
            labels[input_ids == tokenizer.pad_token_id] = -100  # Ignore padding tokens
            # **Move labels to device1**
            labels = labels.to(device1)

            # Measure latency
            start_time = time.time()
            logits, loss = model_fn(input_ids, attention_mask, labels)
            latency = time.time() - start_time
            latencies.append(latency)

            # Accumulate loss
            if loss is not None:
                total_loss += loss.item() * input_ids.size(0)

            # Compute token-level accuracy
            predictions = torch.argmax(logits, dim=-1)
            # Mask to ignore padding tokens
            mask = labels != -100
            correct = (predictions == labels) & mask
            correct_predictions += correct.sum().item()
            total_predictions += mask.sum().item()

            # Update total tokens
            total_tokens += mask.sum().item()

    # Compute average latency and throughput
    avg_latency = sum(latencies) / len(latencies)
    total_time = sum(latencies)
    throughput = total_tokens / total_time  # Tokens per second

    # Compute memory usage
    memory_usage0 = torch.cuda.max_memory_allocated(device0) / (1024 ** 3)
    memory_usage1 = torch.cuda.max_memory_allocated(device1) / (1024 ** 3)
    total_memory_usage = memory_usage0 + memory_usage1

    # Compute perplexity
    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))

    # Compute token-level accuracy
    accuracy = (correct_predictions / total_predictions) * 100

    # Print metrics
    print(f"\nAverage Latency: {avg_latency * 1000:.2f} ms/batch")
    print(f"Throughput: {throughput:.2f} tokens/second")
    print(f"Memory Usage: {total_memory_usage:.2f} GB")
    print(f"Perplexity: {perplexity.item():.2f}")
    print(f"Token-Level Accuracy: {accuracy:.2f}%")


inference(model_parallel_inference_with_loss, valid_loader)