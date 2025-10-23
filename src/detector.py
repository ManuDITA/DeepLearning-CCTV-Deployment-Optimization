from ultralytics import YOLO
import cv2
import glob
import os

from configs import configs

input_dir = configs.SAVE_PATH
output_dir = configs.TRACKED_PATH
os.makedirs(output_dir, exist_ok=True)

model = YOLO("yolov8n.pt")  # small pretrained model

for img_path in sorted(glob.glob(os.path.join(input_dir, "cam_*_frame_*.jpg"))):
    results = model(img_path)
    result_img = results[0].plot()  # draw bounding boxes
    save_path = os.path.join(output_dir, os.path.basename(img_path).replace(".jpg", "_tracked.jpg"))
    cv2.imwrite(save_path, result_img)
    print("Processed:", save_path)