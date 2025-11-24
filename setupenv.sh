#!/bin/bash

# --- 1. Activate the Python environment ---
echo "Activating CARLA Python environment..."
source /home/manu/carla_env/bin/activate

# --- 2. Run CARLA server in the background ---
echo "Starting CARLA simulator (Vulkan)..."
../Carla/CarlaUE4.sh -vulkan -quality-level=Epic &

CARLA_PID=$!
echo "CARLA running with PID: $CARLA_PID"

cleanup() {
    echo ""
    echo "Caught Ctrl+C — shutting down CARLA (PID $CARLA_PID)..."
    
    # Kill CARLA and all its children
    kill -9 $CARLA_PID 2>/dev/null
    
    echo "CARLA terminated."
    exit 0
}

trap cleanup INT

# --- 3. Wait for CARLA to finish loading ---
echo "Waiting for CARLA to start..."
while ! (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null; do
    sleep 1
done

echo "Carla has started"

# Optionally: wait until port 2000 responds
# echo "Waiting for CARLA port 2000..."
# while ! nc -z localhost 2000; do sleep 1; done

# --- 4. Load the map ---
echo "Loading Town01..."
python3 ../Carla/PythonAPI/util/config.py --map Town01


# --- Keep script.running until interrupted ---
echo "CARLA server running. Press Ctrl+C to stop."

# Wait forever until Ctrl+C triggers the trap
while true; do
    sleep 1
done