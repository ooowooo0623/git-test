import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =========================
# 設定
# =========================
SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# =========================
# 隨機種子與 deterministic 設定
# =========================
#def set_seed(seed):
#    os.environ["PYTHONHASHSEED"] = str(seed)
#    random.seed(seed)
#    np.random.seed(seed)
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False
#
#set_seed(SEED)

from torchvision.datasets import USPS

# 資料前處理：USPS 是 16x16 灰階 → 轉成 3 通道並 resize 到 227x227
transform_usps = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.Grayscale(num_output_channels=3),  # 將灰階轉成 RGB
    transforms.ToTensor(),
])

# 載入 USPS 資料集
train_dataset = USPS(root='./data', train=True, download=True, transform=transform_usps)
test_dataset = USPS(root='./data', train=False, download=True, transform=transform_usps)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
#LBP#
class LBPPooling(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, dropout=0.3):
        super(LBPPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.padding = kernel_size // 2
        self.bn = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.downsample = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # LBP weights: 8-bit binary pattern
        self.register_buffer('weights', torch.tensor([1, 2, 4, 8, 16, 32, 64, 128]).view(1, 1, 8, 1, 1))

    def forward(self, x):
        B, C, H, W = x.size()
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Extract 3x3 patches
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)  # [B, C*9, H*W]
        x_unfold = x_unfold.view(B, C, 9, H, W)

        center = x_unfold[:, :, 4]  # 中心像素
        neighbors = torch.cat([x_unfold[:, :, i].unsqueeze(2) for i in range(9) if i != 4], dim=2)  # 8 neighbors

        # Compare neighbors to center
        lbp = (neighbors >= center.unsqueeze(2)).float()  # binary pattern
        lbp = lbp * self.weights  # apply weights
        lbp = lbp.sum(dim=2)  # [B, C, H, W], value in [0, 255]

        lbp = lbp / 255.0  # normalize to [0, 1]
        lbp = self.bn(lbp)
        lbp = self.dropout(lbp)
        lbp = self.downsample(lbp)  # downsample spatially

        return lbp

#mixed#
class MixedPooling(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=0, alpha=0.5):
        super(MixedPooling, self).__init__()
        self.alpha = alpha  # 混合比例：0.5 表示 Max 和 Avg 各一半
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.avgpool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        max_out = self.maxpool(x)
        avg_out = self.avgpool(x)
        return self.alpha * max_out + (1 - self.alpha) * avg_out
#stohastic#
class StochasticPooling2d(nn.Module):
    def __init__(self, kernel_size=3, stride=2):
        super(StochasticPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # unfold 將每個 pooling window 展開成一個 patch
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        patches = patches.transpose(1, 2)  # [B, num_patches, K*K*C]

        # reshape 成 [B, num_patches, C, K*K]
        B, N, D = patches.shape
        C = x.shape[1]
        K = self.kernel_size * self.kernel_size
        patches = patches.view(B, N, C, K)

        # softmax over K
        probs = F.softmax(patches, dim=-1)
        # sample index
        idx = torch.multinomial(probs.view(-1, K), num_samples=1).squeeze(-1)
        # gather sampled values
        selected = patches.view(-1, K).gather(1, idx.unsqueeze(1)).view(B, N, C)

        # reshape back to image
        H_out = (x.shape[2] - self.kernel_size) // self.stride + 1
        W_out = (x.shape[3] - self.kernel_size) // self.stride + 1
        out = selected.permute(0, 2, 1).contiguous().view(B, C, H_out, W_out)
        return out


# =========================
# AlexNet 模型定義
# =========================
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            LBPPooling(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            LBPPooling(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            LBPPooling(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =========================
# 模型初始化與訓練設定
# =========================
model = AlexNet(num_classes=10).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# =========================
# 訓練迴圈
# =========================
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training Loss: {avg_loss:.4f}")

    # =========================
    # 驗證準確率
    # =========================
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")

