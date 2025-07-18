import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.svm import LinearSVC
from skimage import io
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

# ---------- Config ----------
IMG_DIR = '/mnt/nvme_disk2/User_data/nb57077k/.cache/kagglehub/datasets/youssefelebiary/household-trash-recycling-dataset/versions/3/images/train'
LBL_DIR = '/mnt/nvme_disk2/User_data/nb57077k/.cache/kagglehub/datasets/youssefelebiary/household-trash-recycling-dataset/versions/3/labels/train'
CLASS_NAMES = ['cat', 'dog']  # update this
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- Pretrained CNN ----------
vgg16 = models.vgg16(pretrained=True).features.to(DEVICE).eval()
resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_cnn_feature(region):
    region = resize(region).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feature = vgg16(region).cpu().view(-1).numpy()
    return feature

# ---------- IoU ----------
def iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)

# ---------- YOLO Box Reader ----------
def read_yolo_labels(txt_path, w, h):
    boxes = []
    labels = []
    with open(txt_path) as f:
        for line in f.readlines():
            cls, x, y, bw, bh = map(float, line.strip().split())
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls))
    return boxes, labels

# ---------- Selective Search ----------
def get_proposals(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()[:2000]

# ---------- Data Collection ----------
X, Y = [], []

for file in tqdm(os.listdir(IMG_DIR)[:1000]):
    if not file.endswith('.jpg'): continue
    img_path = os.path.join(IMG_DIR, file)
    label_path = os.path.join(LBL_DIR, file.replace('.jpg', '.txt'))

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    gt_boxes, gt_labels = read_yolo_labels(label_path, w, h)
    proposals = get_proposals(img)

    for (x, y, x2, y2) in proposals:
        if x2 - x < 20 or y2 - y < 20: continue
        region = img_rgb[y:y2, x:x2]
        feat = extract_cnn_feature(region)

        assigned = 0  # background by default
        for i, gt_box in enumerate(gt_boxes):
            iou_val = iou([x, y, x2, y2], gt_box)
            if iou_val >= 0.5:
                assigned = gt_labels[i] + 1  # class label + 1
                break

        X.append(feat)
        Y.append(assigned)  # 0 = background

print("Extracted", len(X), "region features")

# ---------- SVM Classifier ----------
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X, Y)

import joblib
joblib.dump(clf, 'rcnn_svm.pkl')
print("RCNN SVM saved as rcnn_svm.pkl")
