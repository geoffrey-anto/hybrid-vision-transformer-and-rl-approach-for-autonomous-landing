#!/usr/bin/env python3

import airsim
import os
import cv2
import time
import numpy as np

import matplotlib.pyplot as plt

def save_lidar_heatmap(point_cloud, filename, grid_size=(100, 100)):
    """
    Convert LiDAR point cloud data into a 2D heatmap.
    - The Z-values are used to generate the heatmap.
    - A colormap is applied, and the heatmap is saved as an image.
    
    Args:
        point_cloud: Flattened LiDAR data [x1, y1, z1, x2, y2, z2, ...]
        filename: Path to save the heatmap image.
        grid_size: Size of the heatmap (pixels).
    """
    if len(point_cloud) < 3:
        print("No LiDAR data to create heatmap.")
        return
    
    # Convert to Nx3 array
    pts = np.array(point_cloud, dtype=np.float32).reshape(-1, 3)
    
    # Extract X, Y, and Z
    x_vals, y_vals, z_vals = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Normalize Z-values for heatmap (scaling between 0 and 255)
    z_min, z_max = np.min(z_vals), np.max(z_vals)
    z_scaled = (z_vals - z_min) / (z_max - z_min) * 255
    z_scaled = z_scaled.astype(np.uint8)

    # Create a 2D histogram (heatmap representation)
    heatmap, _, _ = np.histogram2d(x_vals, y_vals, bins=grid_size, weights=z_scaled)
    
    # Normalize heatmap
    heatmap = np.nan_to_num(heatmap)  # Replace NaNs with zero
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert heatmap to color map
    heatmap_color = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    
    # Save the heatmap image
    cv2.imwrite(filename, heatmap_color)
    print(f"Saved LiDAR heatmap as {filename}")



def compute_landing_score_from_lidar(point_cloud):
    """
    Given a flattened [x1, y1, z1, x2, y2, z2, ...] LiDAR point cloud,
    compute a simple 'ruggedness' measure using standard deviation
    in the Z dimension. Then invert or scale it to produce a landing score.
    """
    # If there are no points, return a default
    if len(point_cloud) < 3:
        return 0.0  # no data => low confidence
    
    # Convert to Nx3 array
    pts = np.array(point_cloud, dtype=np.float32).reshape(-1, 3)
    
    # Calculate standard deviation of the Z-values
    z_std = np.std(pts[:, 2])
    
    # Compute a simple landing score: higher = flatter area
    # You can tweak this formula to match your preference
    # e.g. an exponential or logistic function, etc.
    landing_score = 1.0 / (1.0 + z_std)
    
    return landing_score

def main():
    # --- CONFIGURE THIS SECTION ---
    # Directory where you want to save images
    save_dir = "airsim_data"
    
    # Create folder if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Number of images to capture
    num_images = 1000
    
    # Distance between positions (meters) if you plan to move the drone
    move_step = 5
    
    # Sleep time between moves (seconds)
    move_sleep = 2
    
    # Name of your LiDAR sensor as defined in AirSim settings
    # (Check your settings.json; e.g., "LidarSensor1")
    lidar_name = "LidarSensor1"
    # --- END CONFIGURATION ---
    
    # Create an AirSim MultirotorClient (for a drone). 
    client = airsim.MultirotorClient(ip="192.168.1.3")
    client.confirmConnection()
    
    # If you’re using a drone, enable API control and take off
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    
    # Optional: move to some initial position
    start_position = (0, 0, -6)  # Closer to the ground
    client.moveToPositionAsync(
        start_position[0],
        start_position[1],
        start_position[2],
        2
    ).join()
    for i in range(num_images):
        # Move in a zigzag pattern along x and y axes
        new_x = start_position[0] + i * move_step
        new_y = start_position[1] + (i % 2) * move_step * 2 - move_step
        new_z = start_position[2]
        
        client.moveToPositionAsync(new_x, new_y, new_z, 2).join()
        time.sleep(move_sleep)
        
        # Request a scene image from camera 0
        image_request = airsim.ImageRequest(
            "0",                      # camera_id
            airsim.ImageType.Scene, # scene capture
            False,                  # compress=False -> uncompressed
            False                   # pixels_as_float=False
        )
        
        # Retrieve images
        response = client.simGetImages([image_request])[0]
        
        # Retrieve LiDAR data (point cloud)
        lidar_data = client.getLidarData(lidar_name)
        
        # Compute a landing score based on LiDAR “ruggedness”
        if lidar_data.point_cloud is not None:
            landing_score = compute_landing_score_from_lidar(lidar_data.point_cloud)
        else:
            landing_score = 0.0
        
        if response.width != 0 and response.height != 0:
            # Convert to numpy array
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img_rgba = img1d.reshape(response.height, response.width, 3)
            
            # Convert RGBA to BGR (for OpenCV)
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            
            # Build file name
            filename = os.path.join(save_dir, f"image/{i:03d}.png")
            cv2.imwrite(filename, img_bgr)
            
            heatmap_filename = os.path.join(save_dir, f"lidar/{i:03d}.png")
            save_lidar_heatmap(lidar_data.point_cloud, heatmap_filename)

            print(f"Saved {filename}")
            
            # Get vehicle pose
            vehicle_pose = client.simGetVehiclePose()
            
            # Save metadata (including LiDAR landing score)
            meta_filename = os.path.join(save_dir, f"meta/{i:03d}.txt")
            with open(meta_filename, 'w') as f:
                f.write("Pose:\n")
                f.write(str(vehicle_pose) + "\n")
                f.write("Landing Score (from LiDAR ruggedness):\n")
                f.write(str(landing_score) + "\n")
        else:
            print("Failed to get image!")
    
    # (Optional) Land the drone
    client.landAsync().join()
    
    # Release API control
    client.armDisarm(False)
    client.enableApiControl(False)
    
    print("Data collection complete.")

if __name__ == "__main__":
    main()
