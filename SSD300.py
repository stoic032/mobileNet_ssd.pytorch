import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义SSD模型
class SSD300(nn.Module):
    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        self.num_classes = num_classes  # 目标类别的数量，包括背景类别

        # VGG的基本卷积层部分
        self.base = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ... 更多的层，直到fc7
        ])

        # 额外的卷积层用于提供高级特征图
        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            # ... 更多的层
        ])

        # 定位预测层，预测边界框的偏移量
        self.loc = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            # ... 更多的层，每层输出的数量取决于特征图的大小和锚点数量
        ])

        # 置信度预测层，预测每个类别的置信度
        self.conf = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),
            # ... 更多的层
        ])

    def forward(self, x):
        sources = list()  # 用来保存每个特征图的输出
        loc = list()  # 用来保存边界框位置预测
        conf = list()  # 用来保存类别置信度预测

        # 应用VGG网络的卷积层
        for k in range(23):
            x = self.base[k](x)

        # 保存fc7层的输出
        sources.append(x)

        # 应用额外的卷积层，并保存输出
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # 应用定位和置信度的卷积层
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # 调整预测结果的形状并串联
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # 最终预测结果
        return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)


# 实例化模型
num_classes = 2     # 车牌+背景两个类别
ssd300 = SSD300(num_classes)

# 假设输入图像大小为300x300
x = torch.randn(1, 3, 300, 300)
loc_preds, conf_preds = ssd300(x)

print(loc_preds.shape)  # 预期形状：[batch_size, num_boxes, 4]
print(conf_preds.shape)  # 预期形状：[batch_size, num_boxes, num_classes]
