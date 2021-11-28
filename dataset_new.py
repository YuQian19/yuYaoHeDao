import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torch.utils.data.dataloader as DataLoader
import torchvision
import torchvision.transforms as transforms


# class CustomDataset(Dataset):
#     def __init__(self, label_file_path):
#         with open(label_file_path, 'r') as f:
#             # (image_path(str), image_label(str))
#             self.imgs = list(map(lambda line: line.strip().split(' '), f))
#
#     def __getitem__(self, index):
#         path, label = self.imgs[index]
#         img = transforms.Compose([transforms.ToTensor()])
#         label = int(label)
#         return img, label
#
#     def __len__(self):
#         return len(self.imgs)
#
#
# train_data = CustomDataset('./data/train_data.txt')

transform_train = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    transforms.Resize((756,1008), interpolation=2),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])
train_datasets = torchvision.datasets.ImageFolder('./data/train',transform=transform_train)

show = torchvision.transforms.ToPILImage()

for i, item in enumerate(train_datasets):
    data, label = item
    show(data).show()
    print(data.shape)

