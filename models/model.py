import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
import timm

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(1024 * 1024 * 3, 2)  # 输出为2，对应两个类别

    def forward(self, x):
        x = torch.flatten(x, 1)  # 展平图像
        x = self.fc(x)
        return x

class MyFCNet(nn.Module):
    def __init__(self):
        super(MyFCNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1) # 输出: 16x512x512
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1) # 输出: 32x256x256
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # 输出: 64x128x128
        # 自适应平均池化
        self.pool = nn.AdaptiveAvgPool2d((16, 16)) # 输出: 64x16x16
        # 全连接层
        self.fc1 = nn.Linear(64 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1) # 输出层

    def forward(self, x):
        # 应用卷积层和激活函数
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 应用自适应平均池化
        x = self.pool(x)
        # 展平图像
        x = x.view(x.size(0), -1)
        # 通过全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MyResNet(nn.Module):
    def __init__(self, num_classes=1, pretrained_model_path='pretrained_base_model/resnet50-19c8e357.pth'):
        super(MyResNet, self).__init__()
        # 加载预训练的 ResNet 模型
        self.resnet = models.resnet50()
        self.resnet.load_state_dict(torch.load(pretrained_model_path))

        # 降维层
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 输出: 16x512x512
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 16x256x256
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 输出: 32x128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 输出: 32x64x64
        )
        # 替换 ResNet 的第一个卷积层
        self.resnet.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 替换 ResNet 的全连接层
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.dim_reduce(x)
        return self.resnet(x)

class MyViTModel(nn.Module):
    def __init__(self, num_classes=1, pretrained_model_path='pretrained_base_model/imagenet21k+imagenet2012_ViT-B_16-224.pth'):
        super(MyViTModel, self).__init__()
        # 创建 ViT 模型实例，不加载预训练权重
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)

        # 替换分类器头部
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

        # 添加降维层
        self.resize_conv = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
        self.resize_pool = nn.AdaptiveAvgPool2d((224, 224))

        # 从本地文件加载预训练权重（忽略不匹配的层）
        pretrained_dict = torch.load(pretrained_model_path)
        model_dict = self.vit.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        self.vit.load_state_dict(model_dict)

    def forward(self, x):
        # 应用降维层
        x = self.resize_conv(x)
        x = self.resize_pool(x)
        # 传递给 ViT
        return self.vit(x)