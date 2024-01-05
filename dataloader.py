import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms, ToTensor
from torchvision.datasets import ImageFolder
from PIL import Image


province_list = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

word_list = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    # print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

# if __name__ == '__main__':
#     transform = transforms.Compose([
#         ToTensor(),  # 这个变换将PIL图像转换为PyTorch张量
#     ])
#     train_dataset = ImageFolder(root=r'./dataset/CCPD2020', transform=transform)
#     print(getStat(train_dataset))
#     # ([0.43606845, 0.45841587, 0.42843482], [0.2781826, 0.26929685, 0.27047002])


# 图像预处理
def preprocess(image):
    # 添加batch维度，使图像的形状变为(N, C, H, W)
    # image = image.unsqueeze(0)
    # 调整尺寸为300x300
    # image = F.interpolate(image, size=(300, 300))
    image = F.interpolate(image.unsqueeze(0), size=(300, 300), mode='bilinear', align_corners=False).squeeze(0)

    # 归一化
    normalize = transforms.Normalize(mean=[0.43606845, 0.45841587, 0.42843482],
                                     std=[0.2781826, 0.26929685, 0.27047002])
    image = normalize(image)
    return image


class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, subset, transform=None):
        """
        :param root_dir: 数据集的根目录，包含所有图像。
        :param transform: 在图像上应用的转换。
        """
        self.root_dir = os.path.join(root_dir, subset)
        self.transform = transform
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        _, _, box, points, label, brightness, blurriness = self.images[idx].split('-')

        # 边界框信息
        box = box.split('_')
        box = [list(map(int, i.split('&'))) for i in box]
        box = list(np.array(box).flatten())

        # 车牌框四角的点信息
        points = points.split('_')
        points = [list(map(int, i.split('&'))) for i in points]
        # 将关键点的顺序变为从左上顺时针开始
        points = points[-2:] + points[:2]

        # 读取车牌号
        label = label.split('_')
        # 省份缩写
        province = province_list[int(label[0])]
        # 车牌信息
        words = [word_list[int(i)] for i in label[1:]]
        # 车牌号
        label = province + ''.join(words)

        target = {}
        target["boxes"] = torch.tensor(box)
        target["labels"] = torch.tensor([1])    # 只有绿色车牌

        if self.transform:
            image = self.transform(image)

        return image, target

# 构建transform
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为Tensor
    preprocess                # 应用自定义的预处理函数
])

# 创建数据集实例
train_dataset = LicensePlateDataset(root_dir='./dataset/CCPD2020', subset='train', transform=transform)
val_dataset = LicensePlateDataset(root_dir='./dataset/CCPD2020', subset='val', transform=transform)
test_dataset = LicensePlateDataset(root_dir='./dataset/CCPD2020', subset='test', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)


import matplotlib.pyplot as plt
if __name__ == '__main__':
    for images, targets in train_loader:
        print(images.shape)  # 应该是 [batch_size, channels, height, width]
        # ...

    # for images, targets in train_loader:
    #     print(targets)
        # 获取一个批次的数据
        # images, targets = next(iter(data_loader))

        # # 选择要可视化的图像
        # img_to_show = images[0]  # 选择第一个图像
        #
        # # 选择要可视化的图像并移除批量维度
        # img_to_show = images[0].squeeze(0)  # 选择第一个图像并移除批量维度
        # # 因为图像被标准化了，我们需要去标准化
        # mean = torch.tensor([0.43606845, 0.45841587, 0.42843482])
        # std = torch.tensor([0.2781826, 0.26929685, 0.27047002])
        # img_to_show = img_to_show * std[:, None, None] + mean[:, None, None]
        #
        # # 转换为PIL图像用于展示
        # img_to_show = torchvision.transforms.ToPILImage()(img_to_show)
        #
        # # 使用matplotlib进行图像展示
        # plt.imshow(img_to_show)
        # plt.title("Sample Image")
        # plt.axis('off')
        # plt.show()

