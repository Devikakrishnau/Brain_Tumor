import cv2
import numpy as np
import glob
import torch
from torchvision import models, transforms
from PIL import Image

# Dummy GradCAM function to simulate the backend
def get_dummy_cam(gray_img):
    # Just return a circle in the top right to simulate GradCAM
    cam = np.zeros_like(gray_img, dtype=np.float32)
    cv2.circle(cam, (160, 60), 40, 1.0, -1)
    # Add a halo
    cv2.circle(cam, (160, 60), 80, 0.5, -1)
    return cam

success = 0
for img_path in glob.glob('dataset/Training/meningioma/*.jpg')[:10]:
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    cam = get_dummy_cam(gray)
    cam_mask = (cam > 0.4).astype(np.uint8)
    
    masked_gray = gray * cam_mask
    max_val = np.max(masked_gray)
    
    if max_val > 50:
        _, thresh = cv2.threshold(masked_gray, max_val * 0.7, 255, cv2.THRESH_BINARY)
    else:
        thresh = (cam > 0.85).astype(np.uint8) * 255
        
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        print(f"[{img_path}] Max Val: {max_val}, Tumor Area: {cv2.contourArea(c)}")
        if cv2.contourArea(c) > 10:
            success += 1
    else:
        print(f"[{img_path}] Max Val: {max_val}, Failed to find tumor!")

print(f"Success rate: {success}/10")
