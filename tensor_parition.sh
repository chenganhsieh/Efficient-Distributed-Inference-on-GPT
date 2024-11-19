export MASTER_PORT=29501  # Choose an unused port number
# CUDA_VISIBLE_DEVICES=0,5 torchrun --nproc_per_node=2 tensor_paritioning.py
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 --master_port 60000 tensor_paritioning.py