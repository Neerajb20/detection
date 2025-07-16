import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class YoloDataset(Dataset):
    def __init__(self, image_dir, label_dir, classes, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.classes = classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        label_path = os.path.join(self.label_dir, img_file.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    cls, x_c, y_c, bw, bh = map(float, line.strip().split())
                    # Convert from YOLO (x_center, y_center, w, h) to (x1, y1, x2, y2)
                    x1 = (x_c - bw / 2) * w
                    y1 = (y_c - bh / 2) * h
                    x2 = (x_c + bw / 2) * w
                    y2 = (y_c + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
