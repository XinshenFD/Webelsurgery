#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet50 图片分类模型定义
用于晶状体脱位程度分类（轻/中/重）
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    """ResNet50-based image classifier for eye disease severity."""

    def __init__(self, num_classes=3, input_size=224, pretrained=False):
        super(ResNet50Classifier, self).__init__()
        self.input_size = input_size

        # Load ResNet50 model
        self.resnet = models.resnet50(pretrained=pretrained)

        # Extract ResNet50 feature layers (remove the final fully connected layer)
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])

        # Optimize pooling layer for the given input size
        if input_size == 224:
            pool_size = (7, 7)
            linear_input = 2048 * 7 * 7
        elif input_size <= 512:
            pool_size = (4, 4)
            linear_input = 2048 * 4 * 4
        else:
            pool_size = (7, 7)
            linear_input = 2048 * 7 * 7

        self.adaptive_pool = nn.AdaptiveAvgPool2d(pool_size)

        # Classification head with moderate dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(linear_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
