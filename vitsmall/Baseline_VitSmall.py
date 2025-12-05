import torch
import torch.nn as nn
import timm


class ViTSmallBaseline(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        # Load ViT-Small (pretrained on ImageNet)
        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True
        )

        # Replace head for CIFAR-100
        self.model.head = nn.Linear(384, num_classes)

    def forward(self, x):
        return self.model(x)
