import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model.resnet import NetResDeep
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import time
from GPUtil import showUtilization as gpu_usage
from numba import cuda

data_path = 'data/CIFAR-10/'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_loop(model, train_loader, rank):
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    for epoch in range(1, 100):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(rank), labels.to(rank)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('Epoch {}, Training loss {}'.format(epoch, loss_train / len(train_loader)))
            torch.save(model.module.state_dict(), data_path + 'birds_vs_airplanes.pt') #add module prefix
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"training time: {total_time:.3f} seconds")

def main(rank, world_size):
    setup(rank, world_size)
    cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4915, 0.4823, 0.4468),
                                                    (0.2470, 0.2435, 0.2616))
                                ]))

    sampler = DistributedSampler(cifar10, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(cifar10, batch_size=32, drop_last=False, shuffle=False, sampler=sampler)
    model = NetResDeep().to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    train_loop(model, train_loader, rank)
    destroy_process_group()

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

if __name__ == '__main__':
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
    world_size = torch.cuda.device_count()  
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    # free_gpu_cache()