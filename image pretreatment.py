import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn.utils.rnn import pad_sequence
def custom_collate_fn(batch):
    # 提取图像和目标
    images = [item[0] for item in batch]  # 图像列表
    targets = [item[1] for item in batch]  # 目标列表
    # 堆叠图像
    images = torch.stack(images, dim=0)  # (batch_size, channels, height, width)
    # 初始化批次的目标数据
    batch_bboxes = []
    batch_labels = []
    # 遍历每个样本的目标
    for target in targets:
        # 提取所有目标的 bbox 和 category_id
        bboxes = [t['bbox'] for t in target]  # 每个目标的 bbox 是一个列表 [x, y, width, height]
        labels = [t['category_id'] for t in target]  # 每个目标的 category_id
        # 转换为张量
        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # (num_objects, 4)
        labels = torch.tensor(labels, dtype=torch.int64)  # (num_objects,)
        # 添加到批次中
        batch_bboxes.append(bboxes)
        batch_labels.append(labels)
    # 对边界框和标签进行填充
    # 使用 pad_sequence 填充到相同长度
    batch_bboxes = pad_sequence(batch_bboxes, batch_first=True, padding_value=-1)  # (batch_size, max_num_objects, 4)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-1)  # (batch_size, max_num_objects)
    # 返回堆叠后的图像和目标
    return images, {"boxes": batch_bboxes, "labels": batch_labels}


# 定义数据集的根目录
root = './data'  # 数据集将下载到此目录

# 定义转换（可选）
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),  # 将图像转换为张量
])

# 下载和加载 COCO 2017 数据集
train_dataset = datasets.CocoDetection(
    root=f'{root}/train2017/train2017',  # 数据集根目录
    annFile=f'{root}/annotations_trainval2017/annotations/instances_train2017.json',  # 训练集标注文件
    transform=transform,  # 可选的图像转换
)

val_dataset = datasets.CocoDetection(
    root=f'{root}/val2017/val2017',  # 数据集根目录
    annFile=f'{root}/annotations_trainval2017/annotations/instances_val2017.json',  # 验证集标注文件
    transform=transform,  # 可选的图像转换
)


# 定义 DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,collate_fn=custom_collate_fn)

# 检查数据
# for images, targets in train_loader:
#     print(images.shape)  # 图像张量 (batch_size, channels, height, width)
#     print(targets)       # 目标标注 (list of dictionaries)
#     break
