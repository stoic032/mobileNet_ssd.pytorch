import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


def Depthwise_Separable(in_channels, out_channels, stride, padding):
    """
    深度可分离卷积由两层组成：深度卷积和点卷积。
    mobilenet 使用深度卷积为每个输入通道（输入深度）应用一个过滤器。点卷积是一个简单的 1 × 1 卷积，用于创建深度层输出的线性组合。
    """
    model = nn.Sequential(
        # 3 × 3 深度卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        # 1 × 1 点卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    return model


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, width=1):
        super(MobileNetV1, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.layers = nn.Sequential(
            Depthwise_Separable(width*32, width*64, 1, 1),
            Depthwise_Separable(width*64, width*128, 2, 1),
            Depthwise_Separable(width*128, width*128, 1, 1),
            Depthwise_Separable(width*128, width*256, 2, 1),
            Depthwise_Separable(width*256, width*256, 1, 1),
            Depthwise_Separable(width*256, width*512, 2, 1),
            Depthwise_Separable(width*512, width*512, 1, 1),
            Depthwise_Separable(width*512, width*512, 1, 1),
            Depthwise_Separable(width*512, width*512, 1, 1),
            Depthwise_Separable(width*512, width*512, 1, 1),
            Depthwise_Separable(width*512, width*512, 1, 1),
            Depthwise_Separable(width*512, width*1024, 2, 1),
            Depthwise_Separable(width*1024, width*1024, 2, 4),
        )

        self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(width*1024, num_classes)
        self.softmax = nn.Softmax(dim=1)


class MobileNetV1_SSD300(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV1_SSD300, self).__init__()
        self.num_classes = num_classes

        # 使用MobileNet作为基础模型
        self.base = MobileNetV1(num_classes=self.num_classes)

        # 额外的卷积层用于提供高级特征图
        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        ])
        # 定位预测层，预测边界框的偏移量
        self.loc = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1)
        ])
        # 置信度预测层，预测每个类别的置信度
        self.conf = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        # 定义前向传播
        sources = []
        loc = []
        conf = []

        # 应用MobileNet的子模块
        x = self.base.first_conv(x)
        x = self.base.layers(x)

        # 应用额外的层
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # 应用定位和置信度层
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # print("loc_pred size:", loc.size(0))  # 添加此行来检查尺寸
        return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)


def test(num_classes):
    net = MobileNetV1_SSD300(num_classes)
    y = net(torch.randn(1, 3, 300, 300))    # (batch size, channel, height, width)
    # print(y.size())
    summary(net, (1, 3, 300, 300))


if __name__ == '__main__':
    num_classes = 2  # 绿色车牌 + 背景
    test(num_classes)
