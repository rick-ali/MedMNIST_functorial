'''
Adapted from kuangliu/pytorch-cifar .
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2, get_latent=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.get_latent = get_latent

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        latent = out.view(out.size(0), -1)
        out = self.linear(latent)
        if self.get_latent:
            return out, latent
        return out


def ResNet18(in_channels, num_classes, get_latent=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, get_latent=get_latent)


def ResNet50(in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)

# From https://github.com/QUVA-Lab/partial-escnn/blob/main/networks/networks_cnn.py#L728
# A Probabilistic Approach to Learning the Degree of Equivariance in Steerable CNNs
class CNN3D(torch.nn.Module):
    def __init__(self, n_classes=10, n_channels=1, mnist_type="single", c=6):
        super(CNN3D, self).__init__()

        if mnist_type == "double":
            w = h = d = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif mnist_type == "single":
            w = h = d = 29
            padding_3 = (1, 1, 1)
            padding_4 = (2, 2, 2)
        c = c

        self.upsample = torch.nn.Upsample(size=(h, w, d))

        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv3d(1 * n_channels, c, 7, stride=1, padding=2),
            torch.nn.BatchNorm3d(c),
            torch.nn.ELU(),
        )

        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv3d(c, 2 * c, 5, stride=1, padding=2),
            torch.nn.BatchNorm3d(2 * c),
            torch.nn.ELU(),
        )

        self.pool_1 = torch.nn.AvgPool3d(5, stride=2, padding=1)

        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv3d(2 * c, 4 * c, 3, stride=2, padding=padding_3),
            torch.nn.BatchNorm3d(4 * c),
            torch.nn.ELU(),
        )

        self.pool_2 = torch.nn.AvgPool3d(5, stride=2, padding=1)

        self.block_4 = torch.nn.Sequential(
            torch.nn.Conv3d(4 * c, 6 * c, 3, stride=2, padding=padding_4),
            torch.nn.BatchNorm3d(6 * c),
            torch.nn.ELU(),
        )

        self.block_5 = torch.nn.Sequential(
            torch.nn.Conv3d(6 * c, 6 * c, 3, stride=1, padding=1),
            torch.nn.BatchNorm3d(6 * c),
            torch.nn.ELU(),
        )

        self.pool_3 = torch.nn.AvgPool3d(3, stride=1, padding=0)

        self.block_6 = torch.nn.Conv3d(6 * c, 8 * c, 1)

        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(8 * c),
            torch.nn.ELU(),
            torch.nn.Linear(8 * c, n_classes),
        )

        self.in_type = lambda x: x

    def forward(self, x):
        x = self.upsample(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.pool_1(x)
        x = self.block_3(x)
        x = self.pool_2(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.pool_3(x)
        x = self.block_6(x)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x

    @property
    def network_name(self):
        return "CNN"



# DDMNIST CNN
# From https://github.com/QUVA-Lab/partial-escnn/blob/main/networks/networks_cnn.py#L524
class DDMNISTCNN(torch.nn.Module):
    def __init__(self, n_classes=10, n_channels=1, mnist_type="single", get_latent=False):
        super(DDMNISTCNN, self).__init__()
        self.get_latent = get_latent

        if mnist_type == "double":
            w = h = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif mnist_type == "single":
            w = h = 29
            padding_3 = (1, 1)
            padding_4 = (2, 2)
        c = 6

        self.upsample = torch.nn.Upsample(size=(h, w))
        self.dims = {}

        self.layers = torch.nn.ModuleList([
            self.upsample, # Layer 0 (dim=3249)
            torch.nn.Sequential(
                torch.nn.Conv2d(1 * n_channels, c, 7, stride=1, padding=2),
                torch.nn.BatchNorm2d(c),
                torch.nn.ELU(),
            ), # Layer 1 (dim=18150)
            torch.nn.Sequential(
                torch.nn.Conv2d(c, 2 * c, 5, stride=1, padding=2),
                torch.nn.BatchNorm2d(2 * c),
                torch.nn.ELU(),
            ), # Layer 2 (dim=36300)
            torch.nn.AvgPool2d(5, stride=2, padding=1), # Layer 3 (dim=8748)
            torch.nn.Sequential(
                torch.nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=padding_3),
                torch.nn.BatchNorm2d(4 * c),
                torch.nn.ELU(),
            ), # Layer 4 (dim=5400)
            torch.nn.AvgPool2d(5, stride=2, padding=1), # Layer 5 (dim=1176)
            torch.nn.Sequential(
                torch.nn.Conv2d(4 * c, 6 * c, 3, stride=2, padding=padding_4),
                torch.nn.BatchNorm2d(6 * c),
                torch.nn.ELU(),
            ), # Layer 6 (dim=324)
            torch.nn.Sequential(
                torch.nn.Conv2d(6 * c, 6 * c, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(6 * c),
                torch.nn.ELU(),
            ), # Layer 7 (dim=324)
            torch.nn.AvgPool2d(5, stride=1, padding=1), # Layer 8 (dim=36)
            torch.nn.Conv2d(6 * c, 8 * c, 1), # Layer 9 (dim=48)
        ])

        # final MLP
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(8 * c),
            torch.nn.ELU(),
            torch.nn.Linear(8 * c, n_classes),
        ) # Layer 10 (dim=100)

        self.dims[0] = 3249
        self.dims[1] = 18150
        self.dims[2] = 36300
        self.dims[3] = 8748
        self.dims[4] = 5400
        self.dims[5] = 1176
        self.dims[6] = 324
        self.dims[7] = 324
        self.dims[8] = 36
        self.dims[9] = 48
        self.dims[10] = 100

    def forward(self, x, latent_ids=[-1]):
        latents = []
        for id, layer in enumerate(self.layers):
            x = layer(x)
            if id in latent_ids:
                latents.append(x.reshape(x.shape[0], -1))

        x = x.reshape(x.shape[0], -1)
        out = self.fully_net(x)

        if self.get_latent:
            return out, latents

        return out

    @property
    def network_name(self):
        return "DDMNISTCNN"