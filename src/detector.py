import os
import cv2
import glob
import time
from ultralytics import YOLO
from configs import configs

class DetectorWorker:

    def __init__(self, model_path="yolov8m.pt",
                 output_dir=configs.TRACKED_PATH,
                 input_dir=configs.SAVE_PATH):

        self.model = YOLO(model_path)
        self.output_dir = output_dir
        self.input_dir = input_dir
        
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        print("🔍 Starting detection...")
        start_time = time.time()
        processed = 0
        total_cars_detected = 0
        total_bus_detected = 0
        total_truck_detected = 0

        image_files = sorted(glob.glob(os.path.join(self.input_dir, "cam_*_frame_*.jpg")))
        if not image_files:
            print("⚠️ No images found! Check SAVE_PATH.")
            return 0

        for img_path in image_files:
            results = self.model(img_path, verbose=False)
            
            # Filter only car detections
            car_count = sum(1 for obj in results[0].boxes.cls if (self.model.names[int(obj)] == "car"))
            total_cars_detected += car_count
            bus_count = sum(1 for obj in results[0].boxes.cls if (self.model.names[int(obj)] == "bus"))
            total_bus_detected += bus_count
            truck_count = sum(1 for obj in results[0].boxes.cls if (self.model.names[int(obj)] == "truck"))
            total_truck_detected += truck_count

            # Optional: save annotated image
            result_img = results[0].plot()
            save_path = os.path.join(
                self.output_dir,
                os.path.basename(img_path).replace(".jpg", "_tracked.jpg")
            )
            cv2.imwrite(save_path, result_img)

            processed += 1

        elapsed = time.time() - start_time
        print(f"✅ Detection complete. {processed} images processed in {elapsed:.2f}s "
              f"({processed/elapsed:.1f} img/s),"
              f"total cars detected: {total_cars_detected}, total bus detected: {total_bus_detected}, total trucks detected: {total_truck_detected}")
        return total_cars_detected
