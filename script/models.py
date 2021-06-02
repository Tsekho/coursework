
import torch.nn as nn
import torch.nn.functional as F


"""
ResNet in Pytorch.
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""


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
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion * planes))

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
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
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
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


"""
SimplerNetV1 in Pytorch.
[1] The implementation is basded on:
    https://github.com/D-X-Y/ResNeXt-DenseNet
"""


class SimpleNet(nn.Module):
    def __init__(self, classes=10):
        super(SimpleNet, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(256, classes)
        self.drp = nn.Dropout(0.1)

    def forward(self, x):
        out = self.features(x)

        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = self.drp(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=[3, 3],
                                        stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  64, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.Conv2d(64, 128, kernel_size=[3, 3],
                                        stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  128, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.Conv2d(128, 128, kernel_size=[
                                  3, 3], stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  128, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.Conv2d(128, 128, kernel_size=[
                                  3, 3], stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  128, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),


                              nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                                           dilation=(1, 1), ceil_mode=False),
                              nn.Dropout2d(p=0.1),

                              nn.Conv2d(128, 128, kernel_size=[
                                  3, 3], stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  128, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.Conv2d(128, 128, kernel_size=[
                                  3, 3], stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  128, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.Conv2d(128, 256, kernel_size=[
                                  3, 3], stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  256, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                                           dilation=(1, 1), ceil_mode=False),
                              nn.Dropout2d(p=0.1),

                              nn.Conv2d(256, 256, kernel_size=[
                                  3, 3], stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  256, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.Conv2d(256, 256, kernel_size=[
                                  3, 3], stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  256, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                                           dilation=(1, 1), ceil_mode=False),
                              nn.Dropout2d(p=0.1),

                              nn.Conv2d(256, 512, kernel_size=[
                                  3, 3], stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  512, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                                           dilation=(1, 1), ceil_mode=False),
                              nn.Dropout2d(p=0.1),

                              nn.Conv2d(512, 2048, kernel_size=[
                                  1, 1], stride=(1, 1), padding=(0, 0)),
                              nn.BatchNorm2d(2048, eps=1e-05,
                                             momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.Conv2d(2048, 256, kernel_size=[
                                  1, 1], stride=(1, 1), padding=(0, 0)),
                              nn.BatchNorm2d(
                                  256, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True),

                              nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                                           dilation=(1, 1), ceil_mode=False),
                              nn.Dropout2d(p=0.1),

                              nn.Conv2d(256, 256, kernel_size=[
                                  3, 3], stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(
                                  256, eps=1e-05, momentum=0.05, affine=True),
                              nn.ReLU(inplace=True))

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

        return model


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def SimpleNetV1():
    return SimpleNet()
