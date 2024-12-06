import torch
import torch.nn as nn
import time
import psutil
import os
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# Check for GPU availability and get the number of GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
print(f'Using device: {device}')
print(f'Number of GPUs available: {num_gpus}')

# Load the Penn Treebank dataset
dataset = load_dataset('ptb_text_only', 'penn_treebank')

# Load GPT-2 Medium tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

def tokenize_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['sentence'])

# Convert to PyTorch tensors
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create DataLoader for the validation set
eval_dataset = tokenized_datasets['validation']

# Adjust batch size for better GPU utilization
batch_size = 64 if num_gpus > 1 else 8
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=16)

# Load GPT-2 Medium model
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# Wrap the model with DataParallel if more than one GPU is available
if num_gpus > 1:
    model = nn.DataParallel(model)
model.to(device)
model.eval()  # Set model to evaluation mode

def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    latencies = []
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    total_predictions = 0

    # For memory usage measurement
    process = psutil.Process(os.getpid())

    # Reset GPU peak memory stats
    for i in range(num_gpus):
        torch.cuda.reset_peak_memory_stats(i)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move the entire batch to the main device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Prepare labels for next-token prediction
            labels = input_ids.clone()
            labels[input_ids == tokenizer.pad_token_id] = -100  # Ignore padding tokens

            # Measure latency
            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            latency = time.time() - start_time
            latencies.append(latency)

            # Accumulate loss
            loss = outputs.loss.mean()  # Aggregate loss from all devices
            total_loss += loss.item() * input_ids.size(0)

            # Compute token-level accuracy
            logits = outputs.logits  # [batch_size, seq_length, vocab_size]
            predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_length]

            # Create mask to ignore padding tokens
            mask = labels != -100

            # Count correct predictions
            correct = (predictions == labels) & mask
            correct_predictions += correct.sum().item()
            total_predictions += mask.sum().item()

            # Update total tokens
            total_tokens += mask.sum().item()

    # Compute metrics
    avg_latency = sum(latencies) / len(latencies)
    total_time = sum(latencies)
    throughput = total_tokens / total_time  # Tokens per second
    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))
    accuracy = (correct_predictions / total_predictions) * 100

    # Memory usage in GB
    memory_usage_cpu = process.memory_info().rss / (1024 ** 3)

    # GPU memory usage
    memory_usages_gpu = []
    for i in range(num_gpus):
        memory_usage_gpu = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
        memory_usages_gpu.append(memory_usage_gpu)
    total_memory_usage_gpu = sum(memory_usages_gpu)

    # Print metrics
    print(f"\nPerformance Metrics:")
    print(f"Average Latency per Batch: {avg_latency * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} tokens/second")
    print(f"CPU Memory Usage: {memory_usage_cpu:.2f} GB")
    for i, mem in enumerate(memory_usages_gpu):
        print(f"GPU {i} Memory Usage: {mem:.2f} GB")
    print(f"Total GPU Memory Usage: {total_memory_usage_gpu:.2f} GB")
    print(f"Perplexity: {perplexity.item():.2f}")
    print(f"Token-Level Accuracy: {accuracy:.2f}%")

# Evaluate the model using data parallelism
evaluate_model(model, eval_dataloader, tokenizer, device)
