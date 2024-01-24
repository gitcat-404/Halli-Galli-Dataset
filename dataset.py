import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_file, transform = transforms.ToTensor()):
        self.img_dir = img_dir
        self.transform = transform
        # 读取所有标签
        with open(label_file, 'r') as file:
            self.labels = list(map(int, file.read().split()))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        # 图像文件名与索引相对应
        img_path = os.path.join(self.img_dir, f"{idx}.jpg")
        image = Image.open(img_path).convert('RGB')
        # 获取对应的标签
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label