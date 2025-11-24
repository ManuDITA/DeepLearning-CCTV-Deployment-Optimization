# configs/configs.py

# CARLA connection
CARLA_HOST = "localhost"
CARLA_PORT = 2000

# Environment settings
SIMULATION_FRAMERATE = 30

# Camera settings
CAMERA_IMAGE_WIDTH = 1024
CAMERA_IMAGE_HEIGHT = 1024
CAMERA_FOV = 70
CAMERA_SENSOR_FRAMERATE = 10 #defines after how many frames a camera sensor activates

# Output path
SAVE_PATH = "data/output/test_images/raw"
TRACKED_PATH = "data/output/test_images/tracked"

# Runner configs
NUM_VEHICLES = 50
NUM_CAMERAS = 10
EXPERIMENT_DURATION = 30 #in seconds