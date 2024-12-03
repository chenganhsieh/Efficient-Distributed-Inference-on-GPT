import torch
import math
import time
from torch import nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from transformers.models.gpt2.modeling_gpt2 import Conv1D


device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
devices = [device0, device1]

def load_dataset_and_prepare(tokenizer):
    # Load the validation split of the Penn Treebank dataset
    dataset = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], return_tensors='pt', padding='max_length', max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Create DataLoader
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Create a DataLoader
    valid_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=8)

    # input_ids = torch.stack(valid_encodings['input_ids'])
    # attention_mask = torch.stack(valid_encodings['attention_mask'])
    # valid_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8)

    return valid_loader

def load_model_and_tokenizer():
    # Load the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(devices[0])
    model.eval()  # Set model to evaluation mode

    return model, tokenizer

class PartitionedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, devices=None):
        super(PartitionedLinear, self).__init__()
        if devices is None or len(devices) != 2:
            raise ValueError("Devices must be a list of two torch.device objects.")
        self.in_features = in_features
        self.out_features = out_features
        self.devices = devices
        self.bias = bias

        # Split out_features across devices
        self.out_features_split = [out_features // 2, out_features - out_features // 2]

        # Initialize weights for each partition
        self.weight1 = nn.Parameter(torch.Tensor(self.out_features_split[0], in_features).to(self.devices[0]))
        self.weight2 = nn.Parameter(torch.Tensor(self.out_features_split[1], in_features).to(self.devices[1]))

        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(self.out_features_split[0]).to(self.devices[0]))
            self.bias2 = nn.Parameter(torch.Tensor(self.out_features_split[1]).to(self.devices[1]))
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias1, -bound, bound)
            nn.init.uniform_(self.bias2, -bound, bound)

    def forward(self, input):
        # Perform linear operation on each partition
        input1 = input.to(self.devices[0])
        input2 = input.to(self.devices[1])

        output1 = torch.nn.functional.linear(input1, self.weight1, self.bias1)
        output2 = torch.nn.functional.linear(input2, self.weight2, self.bias2)

        # Move outputs back to the original device and concatenate
        output1 = output1.to(input.device)
        output2 = output2.to(input.device)
        output = torch.cat([output1, output2], dim=-1)
        return output

class PartitionedConv1D(nn.Module):
    def __init__(self, nf, nx, devices):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.devices = devices

        # Split nf (output features) across devices
        split_sizes = [nf // len(devices) for _ in range(len(devices))]
        split_sizes[-1] += nf % len(devices)  # Add remainder to last partition

        self.weight_splits = nn.ParameterList([
            nn.Parameter(torch.empty(nx, split_size, device=device))
            for split_size, device in zip(split_sizes, devices)
        ])
        self.bias_splits = nn.ParameterList([
            nn.Parameter(torch.zeros(split_size, device=device))
            for split_size, device in zip(split_sizes, devices)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight_splits:
            nn.init.normal_(weight, std=0.02)

    def forward(self, x):
        outputs = []
        for weight, bias in zip(self.weight_splits, self.bias_splits):
            x_device = x.to(weight.device)
            size_out = x_device.size()[:-1] + (weight.size(1),)
            output = torch.addmm(bias, x_device.view(-1, x_device.size(-1)), weight)
            output = output.view(size_out)
            outputs.append(output.to(x.device))  # Move output back to original device
        return torch.cat(outputs, dim=-1)
    
def replace_conv1d_with_partitioned_conv1d(module, devices):
    for name, child in module.named_children():
        if isinstance(child, Conv1D):
            # Replace with PartitionedConv1D
            nx = child.weight.shape[0]
            nf = child.weight.shape[1]
            partitioned_conv1d = PartitionedConv1D(nf, nx, devices)

            # Copy weights and biases
            with torch.no_grad():
                weight = child.weight.data
                bias = child.bias.data

                # Split weights and biases
                split_sizes = [nf // len(devices) for _ in range(len(devices))]
                split_sizes[-1] += nf % len(devices)
                indices = [0] + list(torch.cumsum(torch.tensor(split_sizes), dim=0).numpy())
                for i in range(len(devices)):
                    start_idx = indices[i]
                    end_idx = indices[i+1]
                    partitioned_conv1d.weight_splits[i].data.copy_(weight[:, start_idx:end_idx].to(devices[i]))
                    partitioned_conv1d.bias_splits[i].data.copy_(bias[start_idx:end_idx].to(devices[i]))
            setattr(module, name, partitioned_conv1d)
        else:
            replace_conv1d_with_partitioned_conv1d(child, devices)
    
def replace_linear_with_partitioned_linear(module, devices):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace with PartitionedLinear
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None

            partitioned_linear = PartitionedLinear(in_features, out_features, bias=bias, devices=devices)

            # Copy weights and biases
            with torch.no_grad():
                weight = child.weight.data
                bias_data = child.bias.data if bias else None

                # Split weights and biases
                split_sizes = [out_features // 2, out_features - out_features // 2]
                weight1_data = weight[:split_sizes[0], :]
                weight2_data = weight[split_sizes[0]:, :]

                partitioned_linear.weight1.data.copy_(weight1_data.to(devices[0]))
                partitioned_linear.weight2.data.copy_(weight2_data.to(devices[1]))

                if bias:
                    bias1_data = bias_data[:split_sizes[0]]
                    bias2_data = bias_data[split_sizes[0]:]
                    partitioned_linear.bias1.data.copy_(bias1_data.to(devices[0]))
                    partitioned_linear.bias2.data.copy_(bias2_data.to(devices[1]))

            setattr(module, name, partitioned_linear)
        else:
            replace_linear_with_partitioned_linear(child, devices)

def model_inference_with_loss(model, input_ids, attention_mask=None):
    # Move inputs to device0
    input_ids = input_ids.to(devices[0])
    attention_mask = attention_mask.to(devices[0]) if attention_mask is not None else None
    labels = input_ids.clone().to(devices[0])
    labels[input_ids == tokenizer.pad_token_id] = -100 

    # Forward pass through segment 2
    output = model(input_ids, attention_mask=attention_mask, labels = labels)
    logits = output.logits
    loss = output.loss
    return logits, loss
    # Compute loss if labels are provided
    # if labels is not None:
    #     loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()  # Shift labels to align with logits
    #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #     return logits, loss
    # else:
    #     return logits, None

def inference(model, dataloader, tokenizer):
    latencies = []
    total_loss = 0
    total_tokens = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            print(f"Batch:{idx+1}/{len(dataloader)}", end="\r")
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Prepare labels
            labels = input_ids.clone()
            labels[input_ids == tokenizer.pad_token_id] = -100  # Ignore padding tokens
            labels = labels.to(devices[0])

            # Measure latency
            start_time = time.time()
            logits, loss = model_inference_with_loss(model, input_ids, attention_mask)
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


# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()
# Load dataset and prepare dataloader
dataloader = load_dataset_and_prepare(tokenizer)
# replace_conv1d_with_partitioned_conv1d(model, devices)
replace_linear_with_partitioned_linear(model, devices)
inference(model, dataloader, tokenizer)
