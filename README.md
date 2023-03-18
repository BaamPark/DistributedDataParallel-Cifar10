# Distributed Data Parallel Training with PyTorch
This repository contains Python scripts for training a PyTorch model using Distributed Data Parallel (DDP). 
DDP is a parallel and distributed training technique that can significantly reduce training time by leveraging multiple GPUs across multiple nodes. 
The provided Python script demonstrates how to set up and use DDP for training a defined ResNet model on CIFAR10 dataset.

Note: the scope of this project is training a model on local GPUs.

## Code Explanation
`def setup(rank, world_size)`: - Defines the setup function, which sets up the distributed environment for each process.

- `os.environ['MASTER_ADDR'] = 'localhost'`: Sets the master node address to 'localhost'.
- `os.environ['MASTER_PORT'] = '12355'`: Sets the master node port to '12355'.
- `dist.init_process_group("nccl", rank=rank, world_size=world_size)`: Initializes the process group for communication using NCCL backend, and assigns the rank and world size to each process.


`def train_loop(model, train_loader, rank)`: Defines the train_loop function, which is the main training loop for each process.
```python
model = NetResDeep().to(rank)
model = DDP(model, device_ids=[rank], output_device=rank)
```
- The images and labels are moved to the device (GPU) corresponding to the process rank.
