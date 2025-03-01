import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.utils import save_image, make_grid
import math
from vit import ViT
import sys
import tqdm
import matplotlib.pyplot as plt
import yaml
import os

CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cifar10(config):
    # transform image to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['image_size'], config['image_size']), antialias=True), 
        # transforms.RandomHorizontalFlip(p=0.5),             
        # transforms.RandomResizedCrop((config['image_size'], config['image_size']), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2, antialias=True),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # download CIFAR10 dataset. Output images are in PIL format
    trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                            download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], 
                                              shuffle=True, num_workers=4, drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='./data/', train=False, 
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'],
                                             shuffle=False, num_workers=4, drop_last=True)
    
    return trainloader, testloader

def main(run_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")
    

    # model config
    config = {
        'image_size': 32,
        'patch_size': 4,
        'num_channels': 3,
        'embedding_dim': 48,
        'n_heads': 6,
        'hidden_dim': 192,
        'num_blocks': 4,
        'dropout': 0.1,
        'num_classes': len(CLASSES),
        'epochs': 1000,
        'lr': 0.001,
        'lr_decay': None,
        'weight_decay': 0.01,
        'batch_size': 128,
        'data_augment': ['AutoAugment']
    }

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    # get dataloader
    trainloader, testloader = load_cifar10(config)

    # model
    model = ViT(config)
    model = model.to(device)

    # loss
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    if config['lr_decay'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])

    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    # training and evaluating on test set loop
    for i in range(config['epochs']):
        # training
        model.train()
        total_train_loss = 0
        total_correct = 0

        for batch in tqdm.tqdm(trainloader):
            # move images and labels to device
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # get output logits and calculate loss
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_train_loss += loss

            # get prediction and number of correct
            predictions = torch.argmax(logits, dim=1)
            total_correct += torch.sum(predictions == labels).item()

            # optimize weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if config['lr_decay'] is not None:
            scheduler.step()

        avg_train_loss = total_train_loss / len(trainloader)
        train_accuracy = total_correct / len(trainloader.dataset)
        
        train_loss_list.append(avg_train_loss.detach().cpu().numpy())
        train_accuracy_list.append(train_accuracy)

        # evalutaion
        model.eval()
        total_test_loss = 0
        total_correct = 0 
        with torch.no_grad():
            for batch in tqdm.tqdm(testloader):
                # move images and labels to device
                images = batch[0].to(device)
                labels = batch[1].to(device)

                # get output logits and calculate loss
                logits = model(images)
                loss = loss_fn(logits, labels)
                total_test_loss += loss

                # get prediction and number of correct
                predictions = torch.argmax(logits, dim=1)
                total_correct += torch.sum(predictions == labels).item() 

            avg_test_loss = total_test_loss / len(testloader)
            test_accuracy = total_correct / len(testloader.dataset)
            
            test_loss_list.append(avg_test_loss.detach().cpu().numpy())
            test_accuracy_list.append(test_accuracy)

        # graph loss
        plt.figure(1)
        plt.plot(train_loss_list, label="train loss")
        plt.plot(test_loss_list, label="test loss")
        plt.legend()
        plt.savefig(os.path.join(run_dir, 'loss'))

        # graph accuracy
        plt.figure(2)
        plt.plot(train_accuracy_list, label="train accuracy")
        plt.plot(test_accuracy_list, label="test accuracy")
        plt.legend()
        plt.savefig(os.path.join(run_dir, 'accuracy'))

        plt.close(1)
        plt.close(2)

        with open(os.path.join(run_dir, 'log.txt'), 'a') as file:
            sys.stdout = file
            print(f"Finish epoch {i}. Train loss: {avg_train_loss:.4f}. Test loss: {avg_test_loss:.4f}. Train Acc: {train_accuracy:.4f}. Test Acc: {test_accuracy:.4f}. lr: {scheduler.get_last_lr() if config['lr_decay'] is not None else config['lr']}")
            sys.stdout = sys.__stdout__
            print(f"Finish epoch {i}. Train loss: {avg_train_loss:.4f}. Test loss: {avg_test_loss:.4f}. Train Acc: {train_accuracy:.4f}. Test Acc: {test_accuracy:.4f}. lr: {scheduler.get_last_lr() if config['lr_decay'] is not None else config['lr']}")

        # save model
        torch.save(model.state_dict(), os.path.join(run_dir, 'model_weights.pt'))

if __name__ == '__main__':
    run_num = sys.argv[1]
    run_dir = f'./runs/run{run_num}'
    os.mkdir(run_dir)
    
    main(run_dir)