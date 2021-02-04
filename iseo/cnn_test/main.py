import argparse

import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms

from tqdm import tqdm
import matplotlib.pyplot as plt
import os, sys
# 프로젝트 root\dirB를 import 참조 경로에 추가
sys.path.append(
os.path.join(os.path.dirname(__file__), 'cnn_test'))

from config import Config
from network import Model
# test

SEED = 42
torch.manual_seed(SEED)

# Config Parsing
def get_config():
    parser = argparse.ArgumentParser(description="Multi-layer perceptron")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    args = parser.parse_args()

    config = Config(
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    return config


# MNIST dataset
def get_mnist(BATCH_SIZE: int):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    DATASET_PATH = "../dataset"
    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(DATASET_PATH + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(DATASET_PATH + '/test', transform=test_transforms)

    train_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    
    return train_iter, test_iter


# Defining Model
def get_network(LEARNING_RATE: float, device: str):
    network = Model(hidden_size=[32, 64]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

    return network, criterion, optimizer


# Print Model Info
def print_modelinfo(model: nn.Module):
    total_params = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            total_params += len(param.reshape(-1))
    print(f"Number of Total Parameters: {total_params:,d}")


# Define help function
def test_eval(model: nn.Module, test_iter, batch_size: int, device: str):
    with torch.no_grad():
        test_loss = 0
        total = 0
        correct = 0
        for batch_img, batch_lab in test_iter:
            X = batch_img.view(-1, 3, 224, 224).to(device)
            Y = batch_lab.to(device)
            y_pred = model(X)
            _, predicted = torch.max(y_pred.data, 1)
            correct += (predicted == Y).sum().item()
            total += batch_img.size(0)
        val_acc = 100 * correct / total
    return val_acc


# Train MLP Model
def train_model(
    model: nn.Module, train_iter, test_iter, EPOCHS: int, BATCH_SIZE: int, device: str
):
#     # Training Phase
#     print_every = 1
#     print("Start training !")

    print_every = 1
    best_accuracy = 0
    print("Start training !")
    # checkpoint_dir = "weights"

    # if os.path.exists(checkpoint_dir):
    #     model = torch.load( f'{checkpoint_dir}/model.pt') 
    #     model.load_state_dict(
    #         torch.load( f'{checkpoint_dir}/model_state_dict.pt')) 
    #     checkpoint = torch.load(f'{checkpoint_dir}/all.tar')  
    # #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
        
    # else:
    #     model = Model(hidden_size=[64, 32, 64]).to(device)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(EPOCHS):
        loss_val_sum = 0
    for batch_img, batch_lab in train_iter:

        X = batch_img.view(-1, 3, 224, 224).to(device)
        Y = batch_lab.to(device)
        
        # Inference & Calculate los
        y_pred = model.forward(X)
        loss = criterion(y_pred, Y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_val_sum += loss
        
    if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
        # accr_val = M.test(x_test, y_test, batch_size)
        loss_val_avg = loss_val_sum / len(train_iter)
        accr_val = test_eval(model, test_iter, BATCH_SIZE)
        print(f"epoch:[{epoch+1}/{EPOCHS}] cost:[{loss_val_avg:.3f}] test_accuracy:[{accr_val:.3f}]")

    # if accr_val > best_accuracy:
    #     if not os.path.exists(checkpoint_dir):
    #         os.mkdir(checkpoint_dir)
    #     print(f"Model saved : acc - {accr_val}")

    #     torch.save(model, f'{checkpoint_dir}/model.pt')  
    #     torch.save(model.state_dict(), 
    #                f'{checkpoint_dir}/model_state_dict.pt')  
    #     torch.save({
    #             'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict()
    #         }, f'{checkpoint_dir}/all.tar')  

    print("Training Done !")

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_title(title)
    return ax

def test_model(model, test_iter, device: str):
    test_iter = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    data_iter = iter(test_iter)
    images, labels = next(data_iter)

    n_sample = 16
    # sample_indices = np.random.choice(len(mnist_test.targets), n_sample, replace=False)
    test_x = images[:n_sample]
    test_y = labels[:n_sample]

    with torch.no_grad():
        model.eval()
        y_pred = model.forward(test_x.view(-1, 3, 224, 224).type(torch.float).to(device))
        model.train()
    
    y_pred = y_pred.argmax(axis=1)

    plt.figure(figsize=(20, 20))

    for idx in range(n_sample):
        ax = plt.subplot(4, 4, idx+1)
        title = f"Predict: {y_pred[idx]}, Label: {test_y[idx]}"
        imshow(test_x[idx], ax, title)

    plt.show()


if __name__ == "__main__":
    print("PyTorch version:[%s]." % (torch.__version__))

    config = get_config()
    print("This code use [%s]." % (config.device))

    train_iter, test_iter = get_mnist(config.BATCH_SIZE)
    print("Preparing dataset done!")

    network, criterion, optimizer = get_network(config.LEARNING_RATE, config.device)
    print_modelinfo(network)

    train_model(
        network, train_iter, test_iter, config.EPOCHS, config.BATCH_SIZE, config.device
    )

    test_model(network, test_iter, config.device)
