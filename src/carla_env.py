import glob
import sys
import os
import random
import shutil
import time
from queue import Queue
import threading
import cv2
import numpy as np

# Add CARLA Python API
egg_path = glob.glob(os.path.expanduser(
    "~/Documents/Carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
))
if egg_path:
    sys.path.append(egg_path[0])
else:
    print("CARLA egg not found!")

import carla
from configs import configs


class CarlaEnvironment:

    def __init__(self, host=configs.CARLA_HOST, port=configs.CARLA_PORT,
                 num_vehicles=50, num_cameras=1):
        print(f"Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []

        self.num_vehicles = num_vehicles
        self.num_cameras = num_cameras

        self.camera_queues = [Queue(maxsize=1) for _ in range(self.num_cameras)]
        self.preview_frames = [None] * self.num_cameras   # for numpy frames used by imshow
        self.preview_raw = [None] * self.num_cameras      # raw bytes storage filled by callback
        self._start_preview_thread()

        # --- Traffic Manager ---
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.global_percentage_speed_difference(-90)

        # --- Simulation Settings ---
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / configs.SIMULATION_FRAMERATE  # 30 FPS
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)

        print(f"Synchronous simulation started at {1/settings.fixed_delta_seconds:.1f} FPS.")


        # Async image saving
        self.image_queue = Queue()
        self._start_image_saver()

        for tl in self.world.get_actors().filter("traffic.traffic_light"):
            print(f"Setting {tl} green")
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)
            
    # ----------------------------------------------------------------


    def spawn_vehicles(self):

        #Spawning only cars
        car_blueprints = [
            bp for bp in self.blueprint_library.filter("vehicle.*")
            if any(name in bp.id for name in ["audi", "bmw", "tesla", "mercedes", "jeep", "citroen", "CARLA Motors", "Mitsubishi", "toyota", "ford", "volkswagen", "dodge", "lincoln", "mini", "nissan", "chevrolet", "seat"])
        ]
        
        print(f"Found {len(car_blueprints)} vehicle blueprints in CARLA.")

        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        spawned = 0

        if not car_blueprints:
            print("No car blueprints found!")
            return

        for i in range(self.num_vehicles):
            
            
            bp = car_blueprints[i%len(car_blueprints)]
            spawn_point = spawn_points[i % len(spawn_points)]

            # Assign a color only if the blueprint supports it
            if bp.has_attribute("color"):
                # Pick a random recommended color
                color = random.choice(bp.get_attribute("color").recommended_values)
                avilableColors =bp.get_attribute("color").recommended_values
                print(f"{bp.id} has the following color: {color}")
                bp.set_attribute("color", color)
            else:
                color = None  # No color attribute available

            vehicle = self.world.try_spawn_actor(bp, spawn_point)


            if vehicle:
                vehicle.set_autopilot(True, self.tm.get_port())
                self.actor_list.append(vehicle)
                spawned += 1


        print(f"Total vehicles spawned: {spawned}")

    # ----------------------------------------------------------------
    def spawn_static_cameras(self):
        """
        Spawn static cameras that capture every 15 simulation frames.
        """
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(configs.CAMERA_IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(configs.CAMERA_IMAGE_HEIGHT))
        camera_bp.set_attribute("fov", str(configs.CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", str(configs.CAMERA_SENSOR_FRAMERATE/configs.SIMULATION_FRAMERATE))  # fully controlled manually in sync mode

        spawn_points = self.world.get_map().get_spawn_points()
        selected_points = random.sample(spawn_points, min(self.num_cameras, len(spawn_points)))

        self.cameras = []

        for i, point in enumerate(selected_points):
            point.location.z += 8.0
            point.rotation.pitch = -15.0
            camera = self.world.spawn_actor(camera_bp, point)
            self.actor_list.append(camera)
            self.cameras.append(camera)
            camera.listen(self.make_camera_callback(i))
            print(f"Spawned static camera {i} at {point.location}")

    # ---------------
    def spawn_dynamic_camera(self, location, rotation, camera_id=None):

        # assign ID automatically
        self.num_cameras = self.num_cameras + 1
        if camera_id is None:
            camera_id = len(self.cameras)

        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(configs.CAMERA_IMAGE_WIDTH))
        camera_bp.set_attribute("image_size_y", str(configs.CAMERA_IMAGE_HEIGHT))
        camera_bp.set_attribute("fov", str(configs.CAMERA_FOV))
        camera_bp.set_attribute(
            "sensor_tick",
            str(configs.CAMERA_SENSOR_FRAMERATE / configs.SIMULATION_FRAMERATE)
        )

        transform = carla.Transform(location, rotation)
        camera = self.world.spawn_actor(camera_bp, transform)
        if camera is None:
            print(f"❌ Failed to spawn camera at {location}")
            return None

        # track it
        self.actor_list.append(camera)
        self.cameras.append(camera)

        camera.listen(self.make_camera_callback(camera_id))

        print(f"📷 Added dynamic camera {camera_id} at {location} rot={rotation}")

        return camera

    # ----------------------------------------------------------------
    def make_camera_callback(self, cam_id):
        """
        Lightweight, thread-safe camera callback.
        Only enqueues the image for async saving (no GUI or numpy ops).
        """
        def callback(image):
            # Enqueue for saving (saver thread will handle image.save_to_disk)
            self.image_queue.put((image, cam_id, image.frame))
        return callback


    # ----------------------------------------------------------------
    def run_simulation(self, duration=30):
        """
        Run the simulation for a fixed duration of simulation time (not wall time),
        without any OpenCV windows. 30 FPS by default.
        """
        fixed_dt = self.world.get_settings().fixed_delta_seconds
        target_frames = int(duration / fixed_dt)
        print(f"Running simulation for {duration}s of sim time "
            f"({target_frames} frames at {1/fixed_dt:.0f} FPS)...")

        frame_count = 0
        start_wall_time = time.time()

        try:
            while frame_count < target_frames:
                self.world.tick()
                frame_count += 1

                # Print simple progress every simulated second
                if frame_count % int(1 / fixed_dt) == 0:
                    elapsed_wall = time.time() - start_wall_time
                    print(f"Sim time: {frame_count * fixed_dt:.1f}s  |  "
                        f"Frames: {frame_count}  |  "
                        f"Real time: {elapsed_wall:.2f}s")

            elapsed_wall = time.time() - start_wall_time
            print(f"✅ Simulation complete: {frame_count} frames "
                f"= {duration:.1f}s sim time "
                f"(took {elapsed_wall:.2f}s real time).")

            # Wait for pending image saves to finish
            self.image_queue.join()
            self.image_queue.put(None)

        except KeyboardInterrupt:
            print("Simulation interrupted by user.")

    # ----------------------------------------------------------------
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
    def _start_preview_thread(self):
        def preview_worker():
            while True:
                for cam_id, q in enumerate(self.camera_queues):
                    if not q.empty():
                        frame = q.get()
                        cv2.imshow(f"Camera {cam_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        t = threading.Thread(target=preview_worker, daemon=True)
        t.start()
        self._preview_thread = t

    # ----------------------------------------------------------------
    def cleanup(self):

        print("Cleaning up actors...")

        # 4) Force-clean any remaining actors in world (vehicles & sensors)
        try:
            for actor in self.world.get_actors().filter('*vehicle*'):
                try:
                    actor.destroy()
                except Exception:
                    pass
            for actor in self.world.get_actors().filter('*sensor*'):
                try:
                    actor.destroy()
                except Exception:
                    pass
        except Exception:
            pass

        self.actor_list = []

        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except Exception:
            pass

        try:
            self.tm.set_synchronous_mode(False)
        except Exception:
            pass

        print("Cleanup complete. Simulation back to async mode.")
