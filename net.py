from torch import nn
import torch

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))

class Connect(nn.Module):
    def __init__(self, convA, convB, in_channels, out_channels, stride) -> None:
        super().__init__()
        self.convA = convA
        self.convB = convB
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        out = self.convA(x)
        out = self.convB(out)

        return nn.ReLU(inplace=True)(self.short_cut(x) + out)


class MyNet(nn.Module):
    def __init__(self, blocks, num_classes, shortcut_level):
        super().__init__()

        self.in_channels = 32
        self.shortcut_level = shortcut_level

        # (3, 32, 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )
        # (32, 32, 32)
        self.conv2 = self.make_layer(64, blocks[0], 1)
        # (64, 32, 32)
        self.conv3 = self.make_layer(128, blocks[1], 2)
        # (128, 16, 16)
        self.conv4 = self.make_layer(256, blocks[2], 2)
        # (256, 8, 8)
        self.conv5 = self.make_layer(512, blocks[3], 2)
        # (512, 4, 4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # (512, 1, 1)
        self.fc = nn.Linear(self.in_channels, num_classes)
        # (10, )

        self.connect1 = Connect(self.conv2, self.conv3, 32, 128, 2)
        self.connect2 = Connect(self.conv4, self.conv5, 128, 512, 4)

        self.connect_all = Connect(self.connect1, self.connect2, 32, 512, 8)

    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        # 1
        if self.shortcut_level == 1:
            out = self.conv2(out)
            out = self.conv3(out)

            out = self.conv4(out)
            out = self.conv5(out)

        # 2
        elif self.shortcut_level == 2:
            out = self.connect1(out)
            out = self.connect2(out)

        # 3
        elif self.shortcut_level == 3:
            out = self.connect_all(out)

        out = self.pool(out)
        out = out.reshape(out.size(0), -1)

        return self.fc(out)

if __name__ == '__main__':
    model = MyNet([2, 2, 2, 2], 10, 1)
    torch.save(model, './model.pt')