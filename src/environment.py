import airsim
import numpy as np
import time
import math
import cv2
import gym

# If you want to use the "gym.Env" style environment, you can import from gym
# and create your own environment that inherits from gym.Env

class AirSimLandingEnv(gym.Env):
    """
    AirSim Environment for autonomous landing with camera + LiDAR integration.
    """
    def __init__(self,
                 ip_address="192.168.1.3",
                 lidar_name="LidarSensor1",
                 camera_name="0",
                 image_type=airsim.ImageType.Scene,
                 target_altitude=1.0,
                 max_episode_steps=200):
        super(AirSimLandingEnv, self).__init__()
        self.ip_address = ip_address
        self.lidar_name = lidar_name
        self.camera_name = camera_name
        self.image_type = image_type
        self.target_altitude = target_altitude
        self.max_episode_steps = max_episode_steps

        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip=self.ip_address)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Action and Observation space definitions (placeholder examples)
        # Action: [pitch, roll, throttle, yaw_rate]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # For observation, we will store the image + LiDAR features together
        # This is highly application specific. For example:
        self.observation_space = gym.spaces.Box(low=-255, high=255, shape=(3, 224, 224), dtype=np.float32)

        # Episode bookkeeping
        self.step_count = 0

        self.reset()

    def reset(self):
        """
        Resets the environment for a new episode.
        """
        self.step_count = 0

        # Reset the drone
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Takeoff to some safe altitude
        self.client.takeoffAsync(timeout_sec=5).join()
        self.client.moveToZAsync(-10, 2).join()  # move to -10 m altitude

        obs = self._get_observation()
        return obs

    def step(self, action):
        """
        Execute one time step within the environment.
        Action is a 4D array: [pitch, roll, throttle, yaw_rate].
        """
        self.step_count += 1

        pitch, roll, throttle, yaw_rate = action
        # Scale or interpret these actions in a meaningful way for AirSim
        # Below is a simplistic interpretation:
        vx = pitch * 5.0  # max 5 m/s forward/back
        vy = roll * 5.0   # max 5 m/s left/right
        vz = throttle * 5.0  # max 5 m/s up/down
        yaw_rate = yaw_rate * 20.0  # degrees/s

        # Apply the control
        self.client.moveByVelocityAsync(vx, vy, vz, 1,
                                        airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)).join()

        # Get observations
        obs = self._get_observation()

        # Calculate reward
        reward, done = self._compute_reward_done()

        return obs, reward, done, {}

    def _get_observation(self):
        """
        Retrieves image from camera and LiDAR data, fuses them if necessary.
        Returns a single fused observation.
        """
        # Get image
        image_response = self.client.simGetImage(self.camera_name, self.image_type)
        if image_response is None:
            # If no image is returned, return zero array
            rgb_img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Convert to numpy array
            img1d = np.frombuffer(image_response, dtype=np.uint8)
            rgb_img = cv2.imdecode(img1d, cv2.IMREAD_COLOR)
            if rgb_img is None:
                rgb_img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Resize the image to 224x224 (ViT input size, for example)
        rgb_img = cv2.resize(rgb_img, (224, 224))

        # Convert to CHW format
        rgb_img = np.transpose(rgb_img, (2, 0, 1))

        # Get LiDAR data
        lidar_data = self.client.getLidarData(lidar_name=self.lidar_name)
        if len(lidar_data.point_cloud) < 3:
            # No points, return zero
            lidar_points = np.array([0, 0, 0], dtype=np.float32)
        else:
            # each triplet (x, y, z)
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            lidar_points = points.reshape(-1, 3)

        # For demonstration, we’ll just compute some summary statistics
        # (like mean altitude or range) from LiDAR
        if lidar_points.shape[0] > 0:
            avg_height = np.mean(lidar_points[:, 2])
        else:
            avg_height = 0.0

        # Optionally fuse this LiDAR statistic into the image or as an extra channel
        # For now, we’ll store it in an extra dimension or keep it separate
        # A common approach: just keep the image for ViT and pass LiDAR stats separately

        return (rgb_img.astype(np.float32), np.array([avg_height], dtype=np.float32))

    def _compute_reward_done(self):
        """
        Computes the reward and checks if the episode is done based on altitude and safe landing logic.
        """
        # Read current altitude
        state = self.client.getMultirotorState()
        altitude = -state.kinematics_estimated.position.z_val  # negative because z is down

        # If altitude is close to target_altitude, positive reward
        # else negative reward
        distance_from_target = abs(altitude - self.target_altitude)

        # Reward shaping (very basic example):
        reward = -distance_from_target

        done = False
        # End episode if we are low enough (e.g., below 0.5m above ground)
        if altitude <= 0.5:
            reward += 100.0  # big reward for safe-ish landing
            done = True

        # Or if we exceed max steps
        if self.step_count >= self.max_episode_steps:
            done = True

        return reward, done

    def close(self):
        """
        Cleanup tasks.
        """
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
