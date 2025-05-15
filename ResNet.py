import torch.nn as nn
from torchvision.models import resnet50
from torch.optim import SGD
import torch


class AnimalResNet(nn.Module):
    def __init__(self, num_classes=90, freeze_mode='partial'):
        """
        Args:
            num_classes: 新任务的类别数
            freeze_mode:
                'full' - 冻结全部卷积层
                'partial' - 解冻layer4（方案B）
                'none' - 全网络微调
        """
        super(AnimalResNet, self).__init__()

        # 加载预训练模型
        self.backbone = resnet50(pretrained=True)

        # 冻结策略选择
        if freeze_mode == 'full':
            # 方案A：完全冻结卷积层
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_mode == 'partial':
            # 方案B：只解冻layer4（最后一个卷积块）
            for name, param in self.backbone.named_parameters():
                if 'layer4' not in name:  # 冻结除layer4外的所有层
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    print(f"解冻层: {name}")  # 调试用
        # freeze_mode='none'时不做冻结

        # 增强分类头（适用于中等规模数据）
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除原始全连接层

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),  # 添加BN层稳定训练
            nn.Dropout(0.3),  # 适度dropout防过拟合
            nn.Linear(num_features, 1024),
            nn.GELU(),  # 比ReLU更平滑
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


