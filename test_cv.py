import cv2
import numpy as np
import glob

success = 0
for img_path in glob.glob('dataset/Training/meningioma/*.jpg')[:10]:
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    _, head_thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    head_contours, _ = cv2.findContours(head_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if head_contours:
        head_c = max(head_contours, key=cv2.contourArea)
        head_mask = np.zeros_like(gray)
        cv2.drawContours(head_mask, [head_c], -1, 255, thickness=cv2.FILLED)
        
        kernel = np.ones((15, 15), np.uint8)
        brain_core_mask = cv2.erode(head_mask, kernel, iterations=1)
        
        _, bright_thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        tumor_candidates = cv2.bitwise_and(bright_thresh, brain_core_mask)
        
        # Clean up noise
        tumor_candidates = cv2.morphologyEx(tumor_candidates, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        
        contours, _ = cv2.findContours(tumor_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            print(f"[{img_path}] Found tumor area:", cv2.contourArea(c))
            if cv2.contourArea(c) > 100:
                success += 1
        else:
            print(f"[{img_path}] Failed to find tumor!")

print(f"Success rate: {success}/10")
