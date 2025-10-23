from .carla_env import CarlaEnvironment
from configs import configs
import sys
import os

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

    env = CarlaEnvironment(num_vehicles=190, num_cameras=3)
    env.spawn_vehicles()
    env.spawn_static_cameras()
    env.run_simulation(duration=30)
    env.cleanup()

if __name__ == "__main__":
    main()