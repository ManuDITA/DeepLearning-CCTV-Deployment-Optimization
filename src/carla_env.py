import glob
import sys
import os
import random
import shutil
import time

# Dynamically add CARLA Python API egg to sys.path
egg_path = glob.glob(os.path.expanduser(
    "~/Documents/Carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
))
if egg_path:
    sys.path.append(egg_path[0])
else:
    print("CARLA egg not found!")

import carla
from configs import configs
from queue import Queue
import threading
import cv2




class CarlaEnvironment:


    def __init__(self, host=configs.CARLA_HOST, port=configs.CARLA_PORT, num_vehicles=50, num_cameras=1):
        print(f"Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []

        self.num_vehicles = num_vehicles
        self.num_cameras = num_cameras


        self.camera_queues = [Queue(maxsize=1) for _ in range(self.num_cameras)]
        self._start_preview_thread()


        # --- Traffic Manager ---
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)      # important for fixed-step mode
        self.tm.global_percentage_speed_difference(-90)  # slightly faster traffic

        self.greenLights()
        # --- Simulation Settings ---
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 30.0   # 30 FPS simulation
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)

        print(f"Synchronous simulation started at {1/settings.fixed_delta_seconds:.1f} FPS.")
        self.image_queue = Queue()
        self._start_image_saver()

    def greenLights(self):
        for tl in self.world.get_actors().filter("traffic.traffic_light*"):
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)

    def _start_image_saver(self):
        def save_worker():
            while True:
                item = self.image_queue.get()
                if item is None:
                    break
                image, cam_id, frame = item
                filename = os.path.join(configs.SAVE_PATH, f"cam_{cam_id}_frame_{frame}.jpg")
                image.save_to_disk(filename)
                self.image_queue.task_done()

        t = threading.Thread(target=save_worker, daemon=True)
        t.start()
        self._image_saver_thread = t

    # ----------------------------------------------------------------
    def spawn_vehicles(self):
        car_blueprints = [
            bp for bp in self.blueprint_library.filter("vehicle.*")
            if any(name in bp.id for name in ["audi", "bmw", "tesla", "mercedes", "toyota", "ford", "volkswagen"])
        ]
        """Spawn vehicles at random locations."""
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        if not car_blueprints:
            print("No car blueprints found!")
            return

        for i in range(self.num_vehicles):
            bp = random.choice(car_blueprints)
            spawn_point = spawn_points[i % len(spawn_points)]
            vehicle = self.world.try_spawn_actor(bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True, self.tm.get_port())
                self.actor_list.append(vehicle)
                print(f"Spawned car {i+1}: {vehicle.type_id}")

        print(f"Total vehicles spawned: {len(self.actor_list)}")

    # ----------------------------------------------------------------
    def run_simulation(self, duration=30):
        """
        Run simulation for a fixed duration of SIMULATION TIME (not wall time).
        duration is in seconds of simulated world time.
        """
        target_frames = int(duration * (1.0 / self.world.get_settings().fixed_delta_seconds))
        print(f"Running simulation for {duration}s of sim time ({target_frames} frames at 30 FPS)...")

        frame_count = 0
        start_wall_time = time.time()

        try:
            while frame_count < target_frames:
                self.world.tick()
                frame_count += 1
                if((frame_count%30)==0):
                    print(f"We're at second {frame_count/30}")

            elapsed_wall = time.time() - start_wall_time
            print(f"✅ Simulation complete: {frame_count} frames = {duration:.1f}s sim time "
                f"(took {elapsed_wall:.2f}s real time).")

        except KeyboardInterrupt:
            print("Simulation interrupted by user.")

    def spawn_static_cameras(self):
        """
        Spawn static cameras that capture every 15 simulation frames.
        """
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(configs.CAMERA_IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(configs.CAMERA_IMAGE_HEIGHT))
        camera_bp.set_attribute("fov", str(configs.CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", "0.0")  # fully controlled manually in sync mode

        spawn_points = self.world.get_map().get_spawn_points()
        selected_points = random.sample(spawn_points, min(self.num_cameras, len(spawn_points)))

        self.cameras = []
        self.frame_interval = 15  # capture every 15 simulation frames
        self.last_capture_frame = [0] * len(selected_points)

        for i, point in enumerate(selected_points):
            point.location.z += 8.0
            point.rotation.pitch = -15.0
            camera = self.world.spawn_actor(camera_bp, point)
            self.actor_list.append(camera)
            self.cameras.append(camera)

            def send_to_preview(image, cam_id=i):
                # Convert CARLA raw image to numpy
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                frame = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)

                # Put the latest frame in the queue (overwrite if needed)
                q = self.camera_queues[cam_id]
                if q.full():
                    _ = q.get()  # remove old frame
                q.put(frame)
                filename = os.path.join(configs.SAVE_PATH,f"cam_{cam_id}_frame_{image.frame}.jpg")
                image.save_to_disk(filename)

            # register callback
            def make_callback(cam_id):
                def callback(image):
                    current_frame = image.frame
                    # only capture if enough frames have passed
                    if current_frame - self.last_capture_frame[cam_id] >= self.frame_interval:
                        self.image_queue.put((image, cam_id, current_frame))
                        self.last_capture_frame[cam_id] = current_frame
                return callback

            camera.listen(make_callback(i))
            #camera.listen(send_to_preview)
            print(f"Spawned static camera {i} at {point.location}")
    # ----------------------------------------------------------------
    def cleanup(self):
        """Cleanly destroy all actors."""
        print("Cleaning up actors...")
        for actor in list(self.actor_list):
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"Failed to destroy actor {actor.id}: {e}")
        self.actor_list.clear()

        # Restore async mode for next runs
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        self.tm.set_synchronous_mode(False)

        self.image_queue.join()
        self.image_queue.put(None)

        print("Cleanup complete. Simulation back to async mode.")


    # ------------------- Preview Thread -------------------
    def _start_preview_thread(self):
        def preview_worker():
            while True:
                for cam_id, q in enumerate(self.camera_queues):
                    if not q.empty():
                        frame = q.get()
                        window_name = f"Camera {cam_id}"
                        cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

        t = threading.Thread(target=preview_worker, daemon=True)
        t.start()
        self._preview_thread = t