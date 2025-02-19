import numpy as np

def fuse_data(image, lidar_points):
    """
    Example function that fuses image + LiDAR data if desired.
    For this template, we handle it simply in the environment,
    but you could expand here for more advanced data fusion.

    image: np.array shape [3, 224, 224]
    lidar_points: np.array shape [N, 3]
    """
    # e.g., project LiDAR onto image plane, or compute additional channels
    # Return fused representation
    return image