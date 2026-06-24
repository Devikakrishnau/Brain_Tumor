import cv2
import numpy as np

class SizeAgent:

    def estimate(self, image_path):

        img = cv2.imread(image_path, 0)

        # threshold tumor region
        _, mask = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return 0

        tumor = max(contours, key=cv2.contourArea)

        # bounding box
        x,y,w,h = cv2.boundingRect(tumor)

        # image width
        image_width = img.shape[1]

        # average brain width
        brain_width_cm = 14

        pixel_to_cm = brain_width_cm / image_width

        tumor_size_cm = max(w,h) * pixel_to_cm

        return round(tumor_size_cm,2)