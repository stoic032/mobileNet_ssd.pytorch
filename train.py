import torch
from torch.optim import SGD
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F
from model import MobileNetV1_SSD300
from dataloader import train_loader, test_loader, val_loader
import matplotlib.pyplot as plt


class MultiBoxLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(MultiBoxLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        loc_pred, _ = predictions  # 忽略置信度预测
        loc_target, _ = targets['boxes'], targets['labels']
        # print(f"loc_pred={loc_pred}")
        # print(f"loc_target={loc_target}")
        # 计算所有边界框的定位损失
        loc_loss = F.smooth_l1_loss(loc_pred, loc_target, reduction='sum')

        # 总损失仅包括定位损失
        total_loss = self.alpha * loc_loss
        total_loss /= loc_target.size(0)  # 标准化损失

        return total_loss


def train(n_epochs, optimizer, criterion):
    # 训练参数
    n_epochs = 10

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        total_batches = len(train_loader)

        for batch_index, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            predicted_locs, predicted_scores = model(images)
            loss = criterion((predicted_locs, predicted_scores), targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # 打印每个batch的进度
            print(
                f"Epoch [{epoch + 1}/{n_epochs}], Batch [{batch_index + 1}/{total_batches}], Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / total_batches
        train_losses.append(avg_train_loss)

        # 验证循环
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}

                predicted_locs, predicted_scores = model(images)
                loss = criterion((predicted_locs, predicted_scores), targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型并移动到设备
    model = MobileNetV1_SSD300(num_classes=2)
    model.to(device)

    # 定义优化器和损失函数
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    criterion = MultiBoxLoss(alpha=1.0).to(device)

    # 开始训练
    train(n_epochs=10, optimizer=optimizer, criterion=criterion)

