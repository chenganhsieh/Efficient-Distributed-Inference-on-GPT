import torch
import torch.nn as nn
import time
import psutil
import os
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the Penn Treebank dataset
dataset = load_dataset('ptb_text_only', 'penn_treebank')

# Load GPT-2 Medium tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

def tokenize_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['sentence'])

# Convert to PyTorch tensors
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create DataLoader for the validation set
eval_dataset = tokenized_datasets['validation']
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

# Load GPT-2 Medium model
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
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

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
            loss = outputs.loss
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
    memory_usage = process.memory_info().rss / (1024 ** 3)

    # Print metrics
    print(f"\nPerformance Metrics:")
    print(f"Average Latency per Batch: {avg_latency * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} tokens/second")
    print(f"Memory Usage: {memory_usage:.2f} GB")
    print(f"Perplexity: {perplexity.item():.2f}")
    print(f"Token-Level Accuracy: {accuracy:.2f}%")

evaluate_model(model, eval_dataloader, tokenizer, device)
