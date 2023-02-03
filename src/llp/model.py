from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig


def make_model(mdl_config: dict | DictConfig) -> nn.Module:
    if mdl_config["model_name"] == "linear":
        return LinearModel(**mdl_config["parameter"])
    elif mdl_config["model_name"] == "mlp":
        return MLPModel(**mdl_config["parameter"])
    elif mdl_config["model_name"] == "lenet":
        return LeNetModel(**mdl_config["parameter"])
    elif mdl_config["model_name"] == "convnet":
        return ConvnetModel(**mdl_config["parameter"])
    elif mdl_config["model_name"] == "resnet":
        return ResNetModel(**mdl_config["parameter"])
    else:
        raise Exception


class LinearModel(nn.Module):
    def __init__(self, num_classes: int = 1, **kargs: Any) -> None:
        super().__init__()
        self.L1 = nn.Linear(28 * 28, num_classes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = data.reshape(-1, 28 * 28)
        x = self.L1(x)
        x = x.reshape(data.shape[0], -1, *x.shape[1:])
        return x


class MLPModel(nn.Module):
    def __init__(self, num_classes: int = 10, **kargs: Any) -> None:
        super().__init__()
        self.L1 = nn.Linear(28 * 28, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.L2 = nn.Linear(300, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.L3 = nn.Linear(300, 300)
        self.bn3 = nn.BatchNorm1d(300)
        self.L4 = nn.Linear(300, 300)
        self.bn4 = nn.BatchNorm1d(300)
        self.L5 = nn.Linear(300, num_classes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = data.reshape(-1, 28 * 28)
        x = F.relu(self.bn1(self.L1(x)))
        x = F.relu(self.bn2(self.L2(x)))
        x = F.relu(self.bn3(self.L3(x)))
        x = F.relu(self.bn4(self.L4(x)))
        x = self.L5(x)
        x = x.reshape(data.shape[0], -1, *x.shape[1:])
        return x


class LeNetModel(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = data.reshape(-1, *data.shape[2:])
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(data.shape[0], -1, *x.shape[1:])
        return x


class ConvnetModel(nn.Module):
    def __init__(
        self, input_channels: int = 3, num_classes: int = 10, dropout_rate: float = 0.25
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.c1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, num_classes)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = data.reshape(-1, *data.shape[2:])
        x = F.leaky_relu(self.bn1(self.c1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.c2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.c3(x)), negative_slope=0.01)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout2d(x, p=self.dropout_rate)

        x = F.leaky_relu(self.bn4(self.c4(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn5(self.c5(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn6(self.c6(x)), negative_slope=0.01)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout2d(x, p=self.dropout_rate)

        x = F.leaky_relu(self.bn7(self.c7(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn8(self.c8(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn9(self.c9(x)), negative_slope=0.01)
        x = F.avg_pool2d(x, kernel_size=x.data.shape[2])
        x = x.view(x.size(0), x.size(1))
        x = self.l_c1(x)
        x = x.reshape(data.shape[0], -1, *x.shape[1:])
        return x


def ResNetModel(**kwargs: Any) -> ResNet:
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_classes: int, depth: int = 32) -> None:
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, "depth should be 6n+2"
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self, block: Bottleneck | BasicBlock, planes: int, blocks: int, stride: int = 1
    ) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = data.reshape(-1, *data.shape[2:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.reshape(data.shape[0], -1, *x.shape[1:])
        return x
