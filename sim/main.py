import airsim
import time
import numpy as np

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
print("Taking off...")
client.takeoffAsync().join()

# Starting position
start_position = client.getMultirotorState().kinematics_estimated.position
start_x = start_position.x_val
start_y = start_position.y_val
start_z = start_position.z_val  # Note: Z is negative when ascending

# First ascend to 30 meters
print("Ascending to 30 meters...")
client.moveToPositionAsync(start_x, start_y, -30, velocity=5).join()
time.sleep(2)  # Brief pause at max altitude

# Path parameters
total_steps = 100
descent_length = 50            # total horizontal distance
start_altitude = 30           # starting from 30 meters
end_altitude = 2              # target altitude (positive value)
jitter_amplitude = 3          # side-to-side jitter (meters)
jitter_frequency = 3 * np.pi  # frequency of jitter wave

# Calculate step-wise motion
x_vals = np.linspace(start_x, start_x + descent_length, total_steps)
z_vals = np.linspace(-start_altitude, -end_altitude, total_steps)  # descending, so z is negative
y_vals = jitter_amplitude * np.sin(np.linspace(0, jitter_frequency, total_steps)) + start_y

# Fly along the path
print("Following curved descent path with jitter...")
for x, y, z in zip(x_vals, y_vals, z_vals):
    client.moveToPositionAsync(x, y, z, velocity=2).join()
    time.sleep(0.05)  # control rate

# Land and disarm
print("Landing...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

print("Mission complete.")
