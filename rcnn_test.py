import os
import cv2
import torch
import joblib
import numpy as np
from torchvision import models, transforms
from tqdm import tqdm

# ---------- Config ----------
TEST_IMG_DIR = '/mnt/nvme_disk2/User_data/nb57077k/.cache/kagglehub/datasets/youssefelebiary/household-trash-recycling-dataset/versions/3/images/train'
CLASS_NAMES = ['background', 'cat', 'dog']  # must include background as 0
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

# ---------- Selective Search ----------
def get_proposals(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()[:2000]

# ---------- Load SVM ----------
clf = joblib.load('rcnn_svm.pkl')
print("Loaded RCNN SVM model")

# ---------- Inference ----------
for file in tqdm(os.listdir(TEST_IMG_DIR)[:2]):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
    img_path = os.path.join(TEST_IMG_DIR, file)

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    proposals = get_proposals(img)

    for (x1, y1, x2, y2) in proposals:
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        region = img_rgb[y1:y2, x1:x2]
        feature = extract_cnn_feature(region)
        pred = clf.predict([feature])[0]
        print(pred)
        if pred > 0:  # if not background
            label = CLASS_NAMES[pred]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow('Detection', img)
    cv2.waitKey(0)  # press any key to view next image

cv2.destroyAllWindows()
