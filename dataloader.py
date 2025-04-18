import os
import cv2
from torch.utils import data
from torchvision import transforms
import scipy.io as spio
import numpy as np
import skimage
import torch
from pathlib import Path

"""Custom Dataset compatible with prebuilt DataLoader."""
class DistortionDataset(data.Dataset):
    def __init__(self, distortedImgDir, flowDir, transform, distortion_type, data_num):
        self.distorted_image_paths = []
        self.displacement_x_paths = []
        self.displacement_y_paths = []
        # 加载畸变图像路径
        for fs in os.listdir(distortedImgDir):
            # 用 Path 分割文件名（示例文件名：000000_barrel.jpg）
            try:
                stem = Path(fs).stem # "000000_barrel"
                parts = stem.split('_') # parts = ["000000", "barrel"]
                if len(parts) != 2:
                    continue
                name_number, dist_type = parts[0], parts[1]
                if dist_type in distortion_type and int(name_number) < data_num:
                    self.distorted_image_paths.append(os.path.join(distortedImgDir, fs))
            except:
                continue
        # 加载位移场路径（x和y）
        for fs in os.listdir(flowDir):
            # 处理位移场文件名（示例文件名：000000_barrel_disp_x.npy）
            try:
                stem = Path(fs).stem # "000000_barrel_disp_x"
                parts = stem.split('_') # parts = ["000000", "barrel", "disp", "x"]
                if len(parts) != 4:
                    continue
                name_number, dist_type, _, direction = parts
                if dist_type in distortion_type and int(name_number) < data_num:
                    if direction == 'x':
                        self.displacement_x_paths.append(os.path.join(flowDir, fs))
                    elif direction == 'y':
                        self.displacement_y_paths.append(os.path.join(flowDir, fs))
            except:
                continue
        # 按数字前缀排序以确保一一对应
        self.distorted_image_paths.sort(key=lambda p: int(Path(p).stem.split('_')[0]))
        self.displacement_x_paths.sort(key=lambda p: int(Path(p).stem.split('_')[0]))
        self.displacement_y_paths.sort(key=lambda p: int(Path(p).stem.split('_')[0]))
        self.transform = transform

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.distorted_image_paths[index]
        displacement_x_path = self.displacement_x_paths[index]
        displacement_y_path = self.displacement_y_paths[index]
        disp_x = np.load(displacement_x_path).astype(np.float32)  # 加载x位移场
        disp_y = np.load(displacement_y_path).astype(np.float32)  # 加载y位移场
        # 转换为 PyTorch 张量并调整维度（例如 [H, W] → [C=1, H, W]）
        disp_x = torch.from_numpy(disp_x).unsqueeze(0)  # 添加通道维度
        disp_y = torch.from_numpy(disp_y).unsqueeze(0)

        distorted_image = cv2.imread(distorted_image_path)
        stem = Path(distorted_image_path).stem
        label_type = stem.split('_')[1]
        # print(label_type)
        label = 0
        #dist_types=['barrel','pincushion','shear','rotate','perspective','wave']
        if (label_type == 'barrel'):
            label = 0
        elif (label_type == 'pincushion'):
            label = 1
        elif (label_type == 'shear'):
            label = 2
        elif (label_type == 'rotate'):
            label = 3
        elif (label_type == 'perspective'):
            label = 4
        elif (label_type == 'wave'):
            label = 5

        if self.transform is not None:
            trans_distorted_image = self.transform(distorted_image)
        else:
            trans_distorted_image = distorted_image

        return trans_distorted_image, disp_x, disp_y, label

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.distorted_image_paths)


def get_loader(distortedImgDir, flowDir, batch_size, distortion_type, data_num):
    """Builds and returns Dataloader."""

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = DistortionDataset(distortedImgDir, flowDir, transform, distortion_type, data_num)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return data_loader