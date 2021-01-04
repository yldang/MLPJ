# encoding = utf-8
"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from model import AlexNet

# define pytorch device - useful for device-agnostic execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model parameters
NUM_EPOCHS = 1 # original paper
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.002
NUM_CLASSES = 10 # 1000 classes for imagenet 2012 dataset
GPU_NUM = 1 # GPUs to use

# data directory
OUTPUT_DIR = "alexnet_data_out"
LOG_DIR = OUTPUT_DIR + "/tblogs" # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + "/models" # model checkpoints
SPLIT_DIR = OUTPUT_DIR + "/split" # split index

# make directory
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# options
pretrained = True
freeze_layer = True
testing_only = False
load_model = False


if __name__ == "__main__":
    # init seed value
    seed = torch.initial_seed()

    # TensorboardX
    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print("TensorboardX summary writer created")

    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES)

    # load pretrained model
    if pretrained:
        alexnet_dict = alexnet.state_dict()
        # print(alexnet_dict.keys())
        alexnet_pretrained = models.alexnet(pretrained=True)
        pretrained_dict = alexnet_pretrained.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in alexnet_dict}
        # print(pretrained_dict.keys())
        pretrained_dict.pop("classifier.6.weight")
        pretrained_dict.pop("classifier.6.bias")
        alexnet_dict.update(pretrained_dict)
        alexnet.load_state_dict(alexnet_dict)
        # print(alexnet_dict.keys())
        print("Load from pretrained")

    # Freeze parameter
    if freeze_layer:
        for name, value in alexnet.named_parameters():
            if (name != "classifier.6.weight") and (name != "classifier.6.bias"):
                value.requires_grad = False
        print("Freeze layer")

    # train on multiple GPUs
    DEVICE_IDS = list(range(GPU_NUM))
    # alexnet = alexnet.to(device)
    if GPU_NUM > 1:
        alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    alexnet = alexnet.to(device)
    print(alexnet)
    print("Network created")

    # data normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if not testing_only:
        # create data loader
        dataloader_train = DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True)

        print("Training dataloader created")

        # create optimizer
        optimizer = optim.SGD(
            params=filter(lambda p: p.requires_grad, alexnet.parameters()),
            lr=LR_INIT,
            momentum=MOMENTUM,
            weight_decay=LR_DECAY
        )
        print("Optimizer created")

        # multiply LR by 1 / 10 after every 30 epochs
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        print("LR Scheduler created")

        # training
        print("Starting training...")
        total_steps = 1
        for epoch in range(NUM_EPOCHS):
            for imgs, classes in dataloader_train:
                imgs, classes = imgs.to(device), classes.to(device)

                # calculate the loss
                output = alexnet(imgs)

                loss = F.cross_entropy(output, classes.squeeze(1).long())

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log the information and add to tensorboard
                if total_steps % 10 == 0:
                    with torch.no_grad():
                        _, preds = torch.max(output, 1)

                        accuracy = torch.sum(preds == classes.squeeze(1)) / preds.size()[0]

                        print("Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}"
                              .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                        tbwriter.add_scalar("loss", loss.item(), total_steps)
                        tbwriter.add_scalar("accuracy", accuracy.item(), total_steps)

                # print out gradient values and parameter average values
                if total_steps % 100 == 0:
                    with torch.no_grad():
                        # print and save the grad of the parameters
                        # also print and save parameter values
                        for name, parameter in alexnet.named_parameters():
                            if parameter.grad is not None:
                                avg_grad = torch.mean(parameter.grad)
                                tbwriter.add_scalar("grad_avg/{}".format(name), avg_grad.item(), total_steps)
                                tbwriter.add_histogram("grad/{}".format(name),
                                                       parameter.grad.cpu().numpy(), total_steps)
                            if parameter.data is not None:
                                avg_weight = torch.mean(parameter.data)
                                tbwriter.add_histogram("weight/{}".format(name),
                                                       parameter.data.cpu().numpy(), total_steps)
                                tbwriter.add_scalar("weight_avg/{}".format(name), avg_weight.item(), total_steps)

                total_steps += 1

            lr_scheduler.step()

        # save final checkpoints
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "alexnet_states_final.pkl")
        state = {
            "total_steps": total_steps,
            "optimizer": optimizer.state_dict(),
            "model": alexnet.state_dict(),
            "seed": seed,
        }
        torch.save(state, checkpoint_path)

    # testing
    print("Starting testing...")

    dataloader_test = DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True)

    print("Testing dataloader created")

    alexnet.eval()

    accuracy = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for imgs, classes in dataloader_test:
        imgs, classes = imgs.to(device), classes.to(device)

        # calculate the loss
        output = alexnet(imgs)

        with torch.no_grad():
            _, preds = torch.max(output, 1)
            accuracy = accuracy + torch.sum(preds == classes.squeeze(1))

    accuracy = accuracy / preds.size()[0]
    print("Total Accuracy: {}".format(accuracy.item()))




