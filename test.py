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
import numpy as np

CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_image(img, true_label, pred_label):
    img = img / 2 + 0.5  # unnormalize if needed
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'True Label: {CLASSES[true_label]} | Predicted: {CLASSES[pred_label]}')
    plt.axis("off")
    plt.savefig("result.png")

def main(run_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")

    # load config
    with open(os.path.join(run_dir, 'config.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    # get test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['image_size'], config['image_size']), antialias=True), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    # load model
    model = ViT(config)
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, 'model_weights.pt')))

    # evalutaion
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            # move images and labels to device
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # get output logits and calculate loss
            logits = model(images)
            # get prediction and number of correct
            predictions = torch.argmax(logits, dim=1)

            plt.figure(figsize=(15, 15))
            for i in range(16):  # Display 16 images from the batch for visualization
                plt.subplot(4, 4, i+1)
                show_image(images[i].cpu(), labels[i].item(), predictions[i].item())
            
            plt.tight_layout()
            plt.show()

            break

if __name__ == '__main__':
    run_num = sys.argv[1]
    run_dir = f'./runs/run{run_num}'    
    main(run_dir)