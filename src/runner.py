from .carla_env import CarlaEnvironment
from .detector import DetectorWorker
from configs import configs
import shutil
import os
import carla


def clean_output_folder(path):
    if os.path.exists(path):
        print(f"Cleaning folder: {path}")
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(path, exist_ok=True)

def main():
    # Clean output folders
    clean_output_folder(configs.SAVE_PATH)
    clean_output_folder(configs.TRACKED_PATH)

    # --- Create environment ---

    env = CarlaEnvironment(
        num_vehicles=configs.NUM_VEHICLES,
        num_cameras=configs.NUM_CAMERAS
    )

    env.spawn_vehicles()
    env.spawn_static_cameras()
    env.spawn_dynamic_camera(
        carla.Location(x=50, y=10, z=12),
        carla.Rotation(pitch=-20, yaw=0)
    )
    env.run_simulation(duration=configs.EXPERIMENT_DURATION)
    env.cleanup()

    detector = DetectorWorker(model_path="yolov8m.pt")
    detector.run()



if __name__ == "__main__":
    main()
