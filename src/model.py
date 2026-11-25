import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention Module."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: (B, C, H, W) -> (B, C)
        y = self.squeeze(x).view(b, c)
        # Excite: (B, C) -> (B, C, 1, 1)
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale original feature maps
        return x * y.expand_as(x)


class SEResidualBlock(nn.Module):
    """Residual Block augmented with Squeeze-and-Excitation Attention."""
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels, reduction=reduction)
        
        # Shortcut connection to match dimensions if stride > 1 or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class SERACNN(nn.Module):
    """
    Lightweight SERA-CNN for Edge Dysgraphia Screening.
    Input: (B, 1, 256, 256) Grayscale Image
    Output: (B, 1) Raw Logit (apply Sigmoid for probability)
    """
    def __init__(self, num_classes=1):
        super(SERACNN, self).__init__()
        
        # Initial Convolutional Stem: 1x256x256 -> 32x128x128
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # SE-Residual Block 1 (Micro-tremors): 32x128x128 -> 64x128x128
        self.block1 = SEResidualBlock(32, 64, stride=1)
        
        # SE-Residual Block 2 (Macro Malformations): 64x128x128 -> 128x64x64
        self.block2 = SEResidualBlock(64, 128, stride=2)
        
        # Global Average Pooling: 128x64x64 -> 128x1x1
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # Quick architecture & parameter verification
    model = SERACNN()
    dummy_input = torch.randn(1, 1, 256, 256)
    output = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape: {output.shape}")
    print(f"Trainable Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")