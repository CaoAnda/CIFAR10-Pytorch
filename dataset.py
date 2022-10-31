import cv2
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torch
from torchvision.transforms import transforms


# 数据增广方法
transform = transforms.Compose([
    # +4填充至36x36
    transforms.Pad(4),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(), 
    # 随机裁剪至32x32
    transforms.RandomCrop(32), 
    # 转换至Tensor
    transforms.ToTensor(),
    #  归一化
#     transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))
])


def get_dataset_loader(dataset_path, batch_size):
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=True,
        transform=transform,
        # download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=False,
        transform=transform,
        # download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

if __name__ == '__main__':
    cifar10path = './cifar10'
    train_loader, test_loader = get_dataset_loader(cifar10path, 4)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    index = 1

    image = images[index].numpy()
    label = labels[index].numpy()
    image = np.transpose(image, (1, 2, 0))
    
    plt.imsave('pic.jpg', image)
    print(label)
