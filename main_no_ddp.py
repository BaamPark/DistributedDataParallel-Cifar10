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
from torchvision import datasets, transforms
import datetime
import time

data_path = '../data/CIFAR-10/'
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

def prepare(batch_size=32, pin_memory=False, num_workers=0):
    cifar10 = datasets.CIFAR10(
    data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))

    train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True)

    print(len(train_loader))
    return train_loader

def training_loop(model, train_loader):
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    for epoch in range(1, 100):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('Epoch {}, Training loss {}'.format(epoch, loss_train / len(train_loader)))
    end_time = time.time()
    total_time = end_time - start_time

    print(f"training time: {total_time:.3f} seconds")

model = NetResDeep().to(device)
train_loader = prepare()
training_loop(model, train_loader)