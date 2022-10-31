from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.models import resnet18
from dataset import get_dataset_loader
from torch import nn
from tqdm import tqdm

# 设置随机种子
import random
seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    dataset_path = './cifar10'
    train_loader, test_loader = get_dataset_loader(dataset_path, batch_size=256)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epoch_num = 2

    model = resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    train_loss = []
    val_loss = []
    train_epochs_loss = []
    val_epochs_loss = []

    for epoch in range(epoch_num):
        model.train()
        for images, labels in tqdm(train_loader, desc='epoch {}'.format(epoch)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        train_epochs_loss.append(np.average(train_loss))
        print('train loss:{}'.format(np.average(train_loss)))
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in tqdm(test_loader, desc='test'):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs.data, 1)

                val_loss.append(loss.item())
                total += labels.size(0)
                correct += (preds == labels).sum().item()
            val_epochs_loss.append(np.average(val_loss))
            print('loss:{}\tacc:{}%'.format(np.average(val_loss), 100 * correct / total))
        
        print()
    
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(train_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss,'-o',label="train_loss")
    plt.plot(val_epochs_loss,'-o',label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.savefig('analys.png')
    plt.show()



