import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import time
import psutil
import os
from tqdm import tqdm


# Load the Penn Treebank dataset
dataset = load_dataset('ptb_text_only', 'penn_treebank')
train_texts = dataset['train']['sentence']
valid_texts = dataset['validation']['sentence']

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples, return_tensors='pt', padding=True, truncation=True, max_length=128)

train_encodings = tokenize_function(train_texts)
valid_encodings = tokenize_function(valid_texts)

# Create DataLoader
train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'])
valid_dataset = torch.utils.data.TensorDataset(valid_encodings['input_ids'], valid_encodings['attention_mask'])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)

# Load teacher model (GPT-2 Medium)
teacher_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
teacher_model.eval()

# Load student model (GPT-2 Small)
student_model = GPT2LMHeadModel.from_pretrained('gpt2')
student_model.train()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model.to(device)
student_model.to(device)

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Compute the distillation loss between student and teacher logits.
    """
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    student_probs = nn.functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    return loss_fn(student_probs, teacher_probs) * (temperature ** 2)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
num_epochs = 3
temperature = 2.0
alpha = 0.5  # Weight for distillation loss
beta = 0.5   # Weight for student loss (next token prediction)

for epoch in range(num_epochs):
    total_loss = 0
    start_time = time.time()
    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)

        # Shift input_ids and labels for next-token prediction
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = tokenizer.eos_token_id

        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # Get student outputs
        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        student_logits = student_outputs.logits
        loss_student = student_outputs.loss

        # Compute distillation loss
        loss_distill = distillation_loss(student_logits, teacher_logits, temperature)

        # Total loss
        loss = alpha * loss_distill + beta * loss_student
        print(f"====================== Batch idx:{idx}, loss:{loss} ======================", end="\r")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = tokenizer.eos_token_id

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.numel()
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    print(f"Perplexity: {perplexity.item():.2f}")
    model.train()

def measure_inference_metrics(model, dataloader):
    model.eval()
    latencies = []
    total_loss = 0
    total_tokens = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)

            # Prepare labels for calculating loss and accuracy
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
            logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]
            predictions = torch.argmax(logits, dim=-1)  # Shape: [batch_size, seq_length]

            # Mask out padding tokens and labels set to -100
            mask = (labels != -100)
            correct = (predictions == labels) & mask
            correct_predictions += correct.sum().item()
            total_predictions += mask.sum().item()

            # Update total tokens
            total_tokens += mask.sum().item()

    # Compute average latency and throughput
    avg_latency = sum(latencies) / len(latencies)
    total_time = sum(latencies)
    throughput = total_predictions / total_time

    # Compute memory usage
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

    # Compute perplexity
    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))

    # Compute token-level accuracy
    accuracy = (correct_predictions / total_predictions) * 100

    # Print metrics
    print(f"Average Latency: {avg_latency * 1000:.2f} ms/query")
    print(f"Throughput: {throughput:.2f} tokens/second")
    print(f"Memory Usage: {memory_usage:.2f} GB")
    print(f"Perplexity: {perplexity.item():.2f}")
    print(f"Token-Level Accuracy: {accuracy:.2f}%")



# evaluate(student_model, valid_loader)
measure_inference_metrics(student_model, valid_loader)