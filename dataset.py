from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torch
from torchvision.transforms import transforms


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomCrop(32), 
    test_transform
])

def get_dataset_loader(dataset_path, batch_size):
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=True,
        transform=transform,
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=False,
        transform=test_transform,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader

if __name__ == '__main__':
    cifar10path = './cifar10'
    train_loader, test_loader = get_dataset_loader(cifar10path, 12)

    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    index = 10

    image = images[index].numpy()
    label = labels[index].numpy()
    image = np.transpose(image, (1, 2, 0))
    
    plt.imsave('pic.jpg', image)
    print(label)
