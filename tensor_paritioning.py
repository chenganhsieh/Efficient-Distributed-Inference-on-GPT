import os
import torch
import torch.distributed as dist
import deepspeed
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))


def setup_distributed():
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    # Get local rank and set device
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    return local_rank, device


def load_model_and_tokenizer(device):
    # Load the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.eval()  # Set model to evaluation mode

    return model, tokenizer

def get_deepspeed_config():
    deepspeed_config = {
        "train_batch_size": 1,  # Not used during inference
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter_bucket_size": 2e8,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_prefetch_bucket_size": 5e7
        },
        "activation_checkpointing": {
            "partition_activations": False
        }
    }
    return deepspeed_config

def initialize_deepspeed(model, config, device):
    # DeepSpeed requires an optimizer, but it's not used during inference
    optimizer = None

    # Initialize the DeepSpeed engine
    # model_engine, optimizer, _, _ = deepspeed.initialize(
    #     model=model,
    #     optimizer=optimizer,
    #     config=config,
    #     model_parameters=None
    # )
    model_engine = deepspeed.init_inference(
        model,
        tensor_parallel={"tp_size": world_size},
        dtype=torch.float,
        replace_with_kernel_inject=True
    )

    return model_engine

def load_dataset_and_prepare(tokenizer):
    # Load the validation split of the Penn Treebank dataset
    dataset = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=8)

    return dataloader

import time
from tqdm import tqdm

def distributed_inference(model_engine, dataloader, tokenizer, device):
    model_engine.eval()
    latencies = []
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Rank {dist.get_rank()} Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            start_time = time.time()
            # Generate output (e.g., next tokens)
            outputs = model_engine.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50,
                do_sample=False
            )
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)
            total_tokens += outputs.numel()

    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies)
    throughput = total_tokens / sum(latencies)
    memory_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Gather metrics from all ranks
    avg_latency_tensor = torch.tensor(avg_latency, device=device)
    throughput_tensor = torch.tensor(throughput, device=device)
    memory_usage_tensor = torch.tensor(memory_usage, device=device)

    dist.reduce(avg_latency_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(throughput_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(memory_usage_tensor, dst=0, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        num_gpus = dist.get_world_size()
        avg_latency = avg_latency_tensor.item() / num_gpus
        throughput = throughput_tensor.item()
        memory_usage = memory_usage_tensor.item()
        print(f"Average Latency: {avg_latency * 1000:.2f} ms/query")
        print(f"Throughput: {throughput:.2f} tokens/second")
        print(f"Memory Usage per GPU: {memory_usage / num_gpus:.2f} GB")


def main():
    # Setup distributed environment
    local_rank, device = setup_distributed()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(device)

    # Get DeepSpeed configuration
    deepspeed_config = get_deepspeed_config()

    # Initialize DeepSpeed engine
    model_engine = initialize_deepspeed(model, deepspeed_config, device)

    # Load dataset and prepare dataloader
    dataloader = load_dataset_and_prepare(tokenizer)

    # Perform distributed inference
    distributed_inference(model_engine, dataloader, tokenizer, device)

    # Cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
