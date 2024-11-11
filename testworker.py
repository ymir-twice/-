from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms
from dataset import myDataset
import numpy as np
 
 
image_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long)) 
])

# 创建数据集对象
trainset = myDataset(idx_path='stats/train-meta.csv', img_dir='data/images/training/', mask_dir='data/annotations/training/', transform=image_transform, mask_transform=mask_transform)

batch_size = 64
 
print(f"num of CPU: {mp.cpu_count()}")
for num_workers in range(2, mp.cpu_count(), 2):  
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    start = time()
    for epoch in range(1, 3):
        for i, img, mask in enumerate(trainloader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))