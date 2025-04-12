import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
import cv2
from PIL import Image
from transformers import ViTModel, ViTConfig
import random
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration parameters
CONFIG = {
    "image_size": 224,  # Input image size for ViT
    "patch_size": 16,   # ViT patch size
    "lidar_points": 256,  # Number of LiDAR points to use
    "hidden_dim": 128,    # Hidden dimension for policy networks
    "learning_rate": 3e-4,
    "gamma": 0.99,        # Discount factor
    "gae_lambda": 0.95,   # GAE lambda parameter
    "clip_param": 0.2,    # PPO clip parameter
    "value_loss_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "ppo_epochs": 10,     # Number of PPO updates per batch
    "batch_size": 64,
    "buffer_size": 2048,  # Experience buffer size
    "total_timesteps": 1000000,
    "checkpoint_interval": 1000,  # Save model every n steps
    "eval_interval": 5000,  # Evaluate model every n steps
    "checkpoint_dir": "./checkpoints",
    "model_name": "drone_landing_model",
    "render": False,      # Whether to render the environment during training
    "debug_level": 1      # 0: No debug, 1: Basic, 2: Verbose
}

# Create checkpoint directory
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

class TerrainGenerator:
    """Generate synthetic terrain for drone landing simulation"""
    
    def __init__(self, size=100, height_range=(-2, 2), roughness=0.5, seed=None):
        """
        Initialize a terrain generator
        
        Args:
            size: Size of the terrain grid (size x size)
            height_range: Min and max height variation
            roughness: Terrain roughness (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.height_range = height_range
        self.roughness = roughness
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Generate base heightmap
        self.terrain = self._generate_terrain()
        
    def _generate_terrain(self):
        """Generate a random terrain using diamond-square algorithm"""
        # Initialize terrain with zeros
        terrain = np.zeros((self.size, self.size))
        
        # Set random corner values
        terrain[0, 0] = np.random.uniform(*self.height_range)
        terrain[0, self.size-1] = np.random.uniform(*self.height_range)
        terrain[self.size-1, 0] = np.random.uniform(*self.height_range)
        terrain[self.size-1, self.size-1] = np.random.uniform(*self.height_range)
        
        # Diamond-square algorithm
        step = self.size - 1
        amplitude = self.roughness * (self.height_range[1] - self.height_range[0])
        
        while step > 1:
            half_step = step // 2
            
            # Diamond step
            for y in range(half_step, self.size - half_step, step):
                for x in range(half_step, self.size - half_step, step):
                    # Average of four corners
                    avg = (terrain[y-half_step, x-half_step] + 
                           terrain[y-half_step, x+half_step] + 
                           terrain[y+half_step, x-half_step] + 
                           terrain[y+half_step, x+half_step]) / 4.0
                    
                    # Add random displacement
                    terrain[y, x] = avg + np.random.uniform(-amplitude, amplitude)
            
            # Square step
            for y in range(0, self.size, half_step):
                for x in range((y + half_step) % step, self.size, step):
                    # Average of four adjacent points
                    count = 0
                    avg = 0
                    
                    if y >= half_step:  # Top
                        avg += terrain[y-half_step, x]
                        count += 1
                    if y + half_step < self.size:  # Bottom
                        avg += terrain[y+half_step, x]
                        count += 1
                    if x >= half_step:  # Left
                        avg += terrain[y, x-half_step]
                        count += 1
                    if x + half_step < self.size:  # Right
                        avg += terrain[y, x+half_step]
                        count += 1
                    
                    avg /= count
                    terrain[y, x] = avg + np.random.uniform(-amplitude, amplitude)
            
            # Reduce amplitude for the next iteration
            step = half_step
            amplitude *= 0.5
        
        return terrain
    
    def get_height(self, x, y):
        """Get terrain height at given x, y coordinates (interpolated)"""
        # Convert world coordinates to terrain grid coordinates
        grid_x = int((x + self.size/2) % self.size)
        grid_y = int((y + self.size/2) % self.size)
        
        # Clamp to terrain boundaries
        grid_x = max(0, min(grid_x, self.size-1))
        grid_y = max(0, min(grid_y, self.size-1))
        
        return self.terrain[grid_y, grid_x]
    
    def get_normal(self, x, y):
        """Get terrain normal vector at given position"""
        # Get heights at neighboring points
        h_center = self.get_height(x, y)
        h_left = self.get_height(x-1, y)
        h_right = self.get_height(x+1, y)
        h_up = self.get_height(x, y-1)
        h_down = self.get_height(x, y+1)
        
        # Calculate tangent vectors
        tangent_x = np.array([2, 0, h_right - h_left])
        tangent_y = np.array([0, 2, h_down - h_up])
        
        # Calculate normal as cross product of tangents
        normal = np.cross(tangent_x, tangent_y)
        
        # Normalize
        normal = normal / np.linalg.norm(normal)
        
        return normal
    
    def get_camera_image(self, drone_pos, direction, fov=60, resolution=(224, 224)):
        """Generate a synthetic camera image from the drone's perspective"""
        x, y, z = drone_pos
        
        # Create a blank RGB image
        image = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)
        
        # Calculate view parameters
        direction = direction / np.linalg.norm(direction)
        up = np.array([0, 0, 1])  # Up vector (Z-axis)
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, direction)
        
        # Convert FOV to radians
        fov_rad = np.radians(fov)
        
        # Cast rays for each pixel
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                # Calculate ray direction
                u = (j / resolution[1] - 0.5) * 2
                v = (i / resolution[0] - 0.5) * 2
                
                # Ray direction in world space
                ray_dir = direction + right * u * np.tan(fov_rad/2) + true_up * v * np.tan(fov_rad/2) * (resolution[0]/resolution[1])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Simple ray-terrain intersection
                if ray_dir[2] < 0:  # Only cast rays pointing downward
                    # Calculate distance to terrain
                    t = -z / ray_dir[2]
                    
                    # Calculate intersection point
                    hit_x = x + ray_dir[0] * t
                    hit_y = y + ray_dir[1] * t
                    
                    # Get terrain height and normal at intersection
                    terrain_height = self.get_height(hit_x, hit_y)
                    if -z + ray_dir[2] * t > terrain_height:  # If we hit the terrain
                        normal = self.get_normal(hit_x, hit_y)
                        
                        # Simple shading
                        light_dir = np.array([0.5, 0.5, -1.0])
                        light_dir = light_dir / np.linalg.norm(light_dir)
                        
                        # Calculate illumination
                        diffuse = max(0, -np.dot(normal, light_dir))
                        ambient = 0.3
                        illumination = min(1.0, ambient + diffuse)
                        
                        # Coloring based on height and normal
                        height_factor = (terrain_height - self.height_range[0]) / (self.height_range[1] - self.height_range[0])
                        
                        # Generate color based on height
                        if height_factor < 0.2:  # Lower areas - darker
                            color = np.array([70, 90, 40])  # Dark green/brown
                        elif height_factor < 0.4:
                            color = np.array([110, 130, 60])  # Green
                        elif height_factor < 0.6:
                            color = np.array([150, 170, 100])  # Light green
                        elif height_factor < 0.8:
                            color = np.array([180, 180, 150])  # Light brown/tan
                        else:  # Higher areas - lighter
                            color = np.array([200, 200, 180])  # Very light tan
                        
                        # Apply illumination
                        color = color * illumination
                        
                        # Add some texture based on position
                        texture = (np.sin(hit_x * 5) * np.cos(hit_y * 5) + 1) * 10
                        color = np.clip(color + texture, 0, 255)
                        
                        image[i, j] = color
                else:
                    # Sky color
                    image[i, j] = [135, 206, 235]  # Sky blue
        
        return image
    
    def get_lidar_points(self, drone_pos, num_points=256, max_range=50.0, noise_level=0.1):
        """Generate synthetic LiDAR point cloud from drone position"""
        x, y, z = drone_pos
        points = []
        
        # Generate rays in a spherical pattern, biased downward
        for _ in range(num_points):
            # Random spherical coordinates with downward bias
            phi = np.random.uniform(0, 2 * np.pi)  # Azimuth
            theta = np.random.beta(2, 1) * np.pi/2 + np.pi/2  # Elevation (downward biased)
            
            # Convert to cartesian direction
            dx = np.sin(theta) * np.cos(phi)
            dy = np.sin(theta) * np.sin(phi)
            dz = -np.cos(theta)  # Negative because we point down
            
            ray_dir = np.array([dx, dy, dz])
            
            # Simple ray-terrain intersection
            if ray_dir[2] < 0:  # Only cast rays pointing downward
                # Calculate distance to terrain
                t = -z / ray_dir[2]
                
                if t > 0 and t < max_range:
                    # Calculate intersection point
                    hit_x = x + ray_dir[0] * t
                    hit_y = y + ray_dir[1] * t
                    
                    # Get terrain height at intersection
                    terrain_height = self.get_height(hit_x, hit_y)
                    hit_z = -t * ray_dir[2]
                    
                    if hit_z >= terrain_height:
                        # Adjust point to terrain height
                        hit_z = terrain_height
                        
                        # Add some noise
                        hit_x += np.random.normal(0, noise_level)
                        hit_y += np.random.normal(0, noise_level)
                        hit_z += np.random.normal(0, noise_level)
                        
                        # Calculate point relative to drone
                        rel_x = hit_x - x
                        rel_y = hit_y - y
                        rel_z = hit_z - (-z)  # Terrain Z minus drone Z
                        
                        points.append([rel_x, rel_y, rel_z])
        
        # If we don't have enough points, duplicate some with small variations
        while len(points) < num_points:
            if len(points) > 0:
                # Duplicate a random point with noise
                idx = np.random.randint(0, len(points))
                point = points[idx].copy()
                point += np.random.normal(0, noise_level, 3)
                points.append(point)
            else:
                # If no terrain points were found, add artificial points
                # representing flat ground
                rel_x = np.random.uniform(-10, 10)
                rel_y = np.random.uniform(-10, 10)
                rel_z = -z
                points.append([rel_x, rel_y, rel_z])
        
        # If we have too many points, sample random subset
        if len(points) > num_points:
            points = random.sample(points, num_points)
            
        return np.array(points, dtype=np.float32)

class ViTDronePolicy(nn.Module):
    """Vision Transformer-based policy network for drone landing"""
    
    def __init__(self, image_size=224, patch_size=16, lidar_points=256, hidden_dim=128, action_dim=4):
        super(ViTDronePolicy, self).__init__()
        
        # ViT configuration for camera image processing
        vit_config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_dim,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=hidden_dim*4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        # ViT model for visual features
        self.vit = ViTModel(vit_config)
        
        # LiDAR processing network
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_points * 3, hidden_dim * 2),  # Each point has x,y,z
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Combined feature processing
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, image, lidar):
        # Process image with ViT
        vit_output = self.vit(pixel_values=image).last_hidden_state[:, 0]  # Use CLS token
        
        # Process LiDAR data
        lidar_flat = lidar.reshape(lidar.shape[0], -1)  # Flatten points
        lidar_features = self.lidar_encoder(lidar_flat)
        
        # Combine features
        combined_features = torch.cat([vit_output, lidar_features], dim=1)
        features = self.combined_layer(combined_features)
        
        # Actor and critic outputs
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value
    
    def act(self, image, lidar):
        with torch.no_grad():
            action_logits, value = self.forward(image, lidar)
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()
    
    def evaluate(self, image, lidar, action):
        action_logits, value = self.forward(image, lidar)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, value

class DroneLandingEnv:
    """Custom environment for drone landing simulation"""
    
    def __init__(self):
        # Action and observation spaces
        # Actions: Move Forward, Backward, Left, Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: Camera image and LiDAR points
        self.observation_space = spaces.Dict({
            'camera': spaces.Box(low=0, high=255, shape=(3, CONFIG["image_size"], CONFIG["image_size"]), dtype=np.uint8),
            'lidar': spaces.Box(low=-100, high=100, shape=(CONFIG["lidar_points"], 3), dtype=np.float32)
        })
        
        # Create terrain
        self.terrain = TerrainGenerator(
            size=100,
            height_range=(-2, 2),
            roughness=0.5,
            seed=42
        )
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 500
        
        # Drone state
        self.drone_pos = np.array([0.0, 0.0, -20.0])  # x, y, z (-z is up)
        self.drone_vel = np.array([0.0, 0.0, 0.0])
        self.drone_direction = np.array([1.0, 0.0, 0.0])  # Initially facing in +x direction
        
        # Physics parameters
        self.gravity = 9.8
        self.max_speed = 5.0
        self.drag_coef = 0.1
        
        # Visualization
        self.fig = None
        self.ax = None
        self.terrain_mesh = None
        
    def reset(self):
        """Reset the environment at the beginning of an episode"""
        # Random starting position
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = -20  # 20m above ground
        
        # Reset drone state
        self.drone_pos = np.array([x, y, z])
        self.drone_vel = np.array([0.0, 0.0, 0.0])
        self.drone_direction = np.array([1.0, 0.0, 0.0])  # Initially facing in +x direction
        
        # Reset episode steps
        self.episode_steps = 0
        
        # Close any existing visualization
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        
        # Get initial observation
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        self.episode_steps += 1
        
        # Previous position for computing reward
        prev_height_above_ground = self._height_above_ground()
        
        # Convert action to drone movement
        if action == 0:  # Forward
            acceleration = np.array([1.0, 0.0, 0.0])
        elif action == 1:  # Backward
            acceleration = np.array([-1.0, 0.0, 0.0])
        elif action == 2:  # Left
            acceleration = np.array([0.0, -1.0, 0.0])
        elif action == 3:  # Right
            acceleration = np.array([0.0, 1.0, 0.0])
        
        # Apply acceleration
        self.drone_vel += acceleration * 0.5
        
        # Apply gravity
        self.drone_vel[2] += self.gravity * 0.1
        
        # Apply drag
        self.drone_vel -= self.drone_vel * self.drag_coef
        
        # Limit velocity
        speed = np.linalg.norm(self.drone_vel)
        if speed > self.max_speed:
            self.drone_vel = self.drone_vel / speed * self.max_speed
        
        # Update position
        self.drone_pos += self.drone_vel * 0.1
        
        # Update direction based on velocity
        if np.linalg.norm(self.drone_vel) > 0.1:
            horizontal_vel = np.array([self.drone_vel[0], self.drone_vel[1], 0])
            if np.linalg.norm(horizontal_vel) > 0.1:
                self.drone_direction = horizontal_vel / np.linalg.norm(horizontal_vel)
        
        # Calculate distance to ground
        height_above_ground = self._height_above_ground()
        
        # Check for collision with ground
        terrain_height = self.terrain.get_height(self.drone_pos[0], self.drone_pos[1])
        if -self.drone_pos[2] <= terrain_height:
            # Set position to ground level
            self.drone_pos[2] = -terrain_height
            
            # Check if landed or crashed
            landing_speed = abs(self.drone_vel[2])
            horizontal_speed = np.linalg.norm(self.drone_vel[:2])
            
            # Successfully landed if speed is low
            print(f"Landing speed: {landing_speed}, Horizontal speed: {horizontal_speed}")
            landed = landing_speed < 2.0 and horizontal_speed < 1.0
            crashed = not landed
        else:
            landed = False
            crashed = False
        
        # Determine if episode is done
        done = landed or crashed or self.episode_steps >= self.max_episode_steps
        
        # Calculate reward
        reward = self._compute_reward(height_above_ground, prev_height_above_ground, landed, crashed)
        
        # Get observation
        obs = self._get_observation()
        
        # Render if needed
        if CONFIG["render"]:
            self.render()
        
        # Info dictionary
        info = {
            'height': height_above_ground,
            'landed': landed,
            'crashed': crashed,
            'velocity': np.linalg.norm(self.drone_vel),
            'position': self.drone_pos.tolist()
        }
        
        return obs, reward, done, info
    
    def _height_above_ground(self):
        """Calculate drone's height above the terrain"""
        terrain_height = self.terrain.get_height(self.drone_pos[0], self.drone_pos[1])
        return -self.drone_pos[2] - terrain_height
    
    def _compute_reward(self, height, prev_height, landed, crashed):
        """Compute reward based on drone state"""
        if crashed:
            return -100  # Large penalty for crashing
        
        if landed:
            return 100  # Large reward for successful landing
        
        # Encourage descending at a reasonable rate
        height_diff = prev_height - height
        height_reward = 0.5 * height_diff  # Reward for descending
        
        # Penalize being far from the ground
        distance_penalty = -0.1 * height
        
        # Penalty for high velocity when close to ground
        velocity_penalty = 0
        if height < 5 and self.drone_vel[2] > 2:
            velocity_penalty = -0.5 * (self.drone_vel[2] - 2)**2
        
        # Encourage the drone to stay level (not tilted)
        horizontal_speed = np.linalg.norm(self.drone_vel[:2])
        if height < 10:
            horizontal_penalty = -0.2 * horizontal_speed
        else:
            horizontal_penalty = 0
        
        return height_reward + distance_penalty + velocity_penalty + horizontal_penalty
    
    def _get_observation(self):
        """Get observation (camera image and LiDAR points)"""
        # Get camera image
        camera_img = self.terrain.get_camera_image(
            self.drone_pos, 
            np.array([self.drone_direction[0], self.drone_direction[1], -0.5]),  # Look downward
            fov=60, 
            resolution=(CONFIG["image_size"], CONFIG["image_size"])
        )
        
        # Get LiDAR points
        lidar_points = self.terrain.get_lidar_points(
            self.drone_pos,
            num_points=CONFIG["lidar_points"],
            max_range=50.0
        )
        
        # Reshape and normalize camera image
        camera_img = np.transpose(camera_img, (2, 0, 1))  # Channel-first format
        
        return {
            'camera': camera_img.astype(np.float32) / 255.0,  # Normalize to [0,1]
            'lidar': lidar_points
        }
    
    def render(self, mode='human'):
        """Render the environment"""
        if self.fig is None:
            # Create figure and axes
            self.fig = plt.figure(figsize=(12, 8))
            self.fig.canvas.manager.set_window_title('Drone Landing Simulation')
            
            # Create subplots
            self.ax = self.fig.add_subplot(111, projection='3d')
            
            # Create terrain mesh
            terrain_size = 50
            x = np.linspace(-terrain_size/2, terrain_size/2, 50)
            y = np.linspace(-terrain_size/2, terrain_size/2, 50)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Fill in terrain heights
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = self.terrain.get_height(X[i, j], Y[i, j])
            
            # Plot terrain
            self.terrain_mesh = self.ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8)
            
            # Set labels and limits
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_xlim(-terrain_size/2, terrain_size/2)
            self.ax.set_ylim(-terrain_size/2, terrain_size/2)
            self.ax.set_zlim(-5, 25)
            self.ax.set_title('Drone Landing Simulation')
        
        # Clear previous drone plot
        for collection in self.ax.collections:
            if collection != self.terrain_mesh:
                collection.remove()
        
        # Plot drone position
        self.ax.scatter(
            self.drone_pos[0], 
            self.drone_pos[1], 
            -self.drone_pos[2],  # Convert to positive z for visualization
            color='red', 
            s=100, 
            marker='o'
        )
        
        # Plot velocity vector
        vel_scale = 2.0
        self.ax.quiver(
            self.drone_pos[0], 
            self.drone_pos[1], 
            -self.drone_pos[2],
            self.drone_vel[0] * vel_scale, 
            self.drone_vel[1] * vel_scale, 
            -self.drone_vel[2] * vel_scale,
            color='blue'
        )
        
        # Plot direction vector
        dir_scale = 5.0
        self.ax.quiver(
            self.drone_pos[0], 
            self.drone_pos[1], 
            -self.drone_pos[2],
            self.drone_direction[0] * dir_scale, 
            self.drone_direction[1] * dir_scale, 
            0,
            color='green'
        )
        
        # Display height above ground
        height = self._height_above_ground()
        self.ax.set_title(f'Drone Landing Simulation - Height: {height:.2f}m')
        
        # Draw and pause to update
        plt.draw()
        plt.pause(0.01)
        
        return self.fig

class Buffer:
    """PPO experience buffer"""
    
    def __init__(self, buffer_size, image_size, lidar_points):
        self.camera_imgs = np.zeros((buffer_size, 3, image_size, image_size), dtype=np.float32)
        self.lidar_data = np.zeros((buffer_size, lidar_points, 3), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        self.size = 0
        self.buffer_size = buffer_size
        
    def add(self, camera_img, lidar_data, action, reward, value, log_prob, done):
        idx = self.size % self.buffer_size
        
        self.camera_imgs[idx] = camera_img
        self.lidar_data[idx] = lidar_data
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        
        self.size += 1
        
    def get(self):
        indices = np.arange(self.buffer_size)
        return (
            torch.FloatTensor(self.camera_imgs),
            torch.FloatTensor(self.lidar_data),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.dones)
        )
    
    def clear(self):
        self.size = 0

def compute_gae(buffer, last_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation"""
    advantages = np.zeros_like(buffer.rewards)
    last_gae = 0
    
    # Reverse iteration for GAE calculation
    for t in reversed(range(len(buffer.rewards))):
        if t == len(buffer.rewards) - 1:
            next_value = last_value
        else:
            next_value = buffer.values[t + 1]
        
        next_non_terminal = 1.0 - buffer.dones[t]
        delta = buffer.rewards[t] + gamma * next_value * next_non_terminal - buffer.values[t]
        
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    # Calculate returns
    returns = advantages + buffer.values
    
    return returns, advantages

def ppo_update(policy, optimizer, buffer, device, config):
    """Update policy using PPO algorithm"""
    # Get buffer data
    camera_imgs, lidar_data, actions, rewards, values, old_log_probs, dones = buffer.get()
    
    # Move data to device
    camera_imgs = camera_imgs.to(device)
    lidar_data = lidar_data.to(device)
    actions = actions.to(device)
    
    # Compute GAE
    with torch.no_grad():
        last_obs = {
            'camera': torch.FloatTensor(buffer.camera_imgs[-1:]).to(device),
            'lidar': torch.FloatTensor(buffer.lidar_data[-1:]).to(device)
        }
        _, last_value = policy(last_obs['camera'], last_obs['lidar'])
        last_value = last_value.cpu().numpy()[0]
    
    returns, advantages = compute_gae(buffer, last_value, config["gamma"], config["gae_lambda"])
    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO mini-batch updates
    total_loss = 0
    total_value_loss = 0
    total_policy_loss = 0
    total_entropy = 0
    
    # Mini-batch training
    batch_size = config["batch_size"]
    n_samples = buffer.buffer_size
    
    for _ in range(config["ppo_epochs"]):
        # Generate random mini-batches
        indices = np.random.permutation(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get mini-batch data
            batch_camera = camera_imgs[batch_indices]
            batch_lidar = lidar_data[batch_indices]
            batch_actions = actions[batch_indices]
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices].to(device)
            
            # Evaluate actions
            log_probs, entropy, values = policy.evaluate(batch_camera, batch_lidar, batch_actions)
            values = values.squeeze()
            
            # Calculate policy loss with clipping
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - config["clip_param"], 1.0 + config["clip_param"]) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = nn.MSELoss()(values, batch_returns)
            
            # Calculate entropy loss
            entropy_loss = -entropy.mean()
            
            # Calculate total loss
            loss = policy_loss + config["value_loss_coef"] * value_loss + config["entropy_coef"] * entropy_loss
            
            # Update the network
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            total_entropy += entropy.mean().item()
    
    n_updates = config["ppo_epochs"] * (n_samples // batch_size + int(n_samples % batch_size > 0))
    
    return {
        'loss': total_loss / n_updates,
        'value_loss': total_value_loss / n_updates,
        'policy_loss': total_policy_loss / n_updates,
        'entropy': total_entropy / n_updates
    }

def evaluate_policy(env, policy, device, num_episodes=5):
    """Evaluate the policy by running a few episodes"""
    returns = []
    successes = 0
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Convert observation to tensor
            camera_img = torch.FloatTensor(obs['camera']).unsqueeze(0).to(device)
            lidar_data = torch.FloatTensor(obs['lidar']).unsqueeze(0).to(device)
            
            # Get action
            with torch.no_grad():
                action, _, _ = policy.act(camera_img, lidar_data)
            
            # Take step
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Check if successfully landed
            if done and info.get('landed', False):
                successes += 1
        
        returns.append(episode_reward)
    
    return {
        'mean_return': np.mean(returns),
        'success_rate': successes / num_episodes
    }

def train():
    """Main training loop"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = DroneLandingEnv()
    
    # Create policy network
    policy = ViTDronePolicy(
        image_size=CONFIG["image_size"],
        patch_size=CONFIG["patch_size"],
        lidar_points=CONFIG["lidar_points"],
        hidden_dim=CONFIG["hidden_dim"]
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(policy.parameters(), lr=CONFIG["learning_rate"])
    
    # Create buffer
    buffer = Buffer(
        buffer_size=CONFIG["buffer_size"],
        image_size=CONFIG["image_size"],
        lidar_points=CONFIG["lidar_points"]
    )
    
    # Training variables
    step = 0
    episode = 0
    best_success_rate = 0
    
    # Logging
    log_data = {
        'steps': [],
        'episodes': [],
        'returns': [],
        'success_rates': [],
        'losses': [],
        'value_losses': [],
        'policy_losses': [],
        'entropies': []
    }
    
    # Main training loop
    obs = env.reset()
    done = False
    episode_reward = 0
    steps_to_land = 0
    
    print("Starting training...")
    while step < CONFIG["total_timesteps"]:
        # Collect data for buffer
        buffer_size = 0
        
        while buffer_size < CONFIG["buffer_size"]:
            if step % 100 == 0 and CONFIG["debug_level"] > 0:
                print(f"Episode {episode}, Step {step}, Buffer size {buffer_size}")
                
            # Convert observation to tensor
            camera_img = torch.FloatTensor(obs['camera']).unsqueeze(0).to(device)
            lidar_data = torch.FloatTensor(obs['lidar']).unsqueeze(0).to(device)
            
            # Get action
            with torch.no_grad():
                action, log_prob, value = policy.act(camera_img, lidar_data)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps_to_land += 1
            
            # Add to buffer
            buffer.add(
                obs['camera'],
                obs['lidar'],
                action,
                reward,
                value,
                log_prob,
                done
            )
            
            obs = next_obs
            buffer_size += 1
            step += 1
            
            # Handle episode termination
            if done:
                # Log episode statistics
                if CONFIG["debug_level"] > 0:
                    landing_status = "LANDED" if info.get('landed', False) else "CRASHED"
                    print(f"Episode {episode} finished: Reward = {episode_reward:.2f}, Status = {landing_status}")
                    print(f"  Steps to land: {steps_to_land}")
                    print(f"  Height above ground: {info['height']:.2f}m")
                
                steps_to_land = 0
                
                # Reset environment
                obs = env.reset()
                episode += 1
                episode_reward = 0
            
            # Checkpoint model
            if step % CONFIG["checkpoint_interval"] == 0:
                checkpoint_path = os.path.join(
                    CONFIG["checkpoint_dir"],
                    f"{CONFIG['model_name']}_{step}.pt"
                )
                torch.save({
                    'step': step,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                
                if CONFIG["debug_level"] > 0:
                    print(f"Saved checkpoint at step {step}")
            
            # Evaluate policy
            if step % CONFIG["eval_interval"] == 0:
                eval_results = evaluate_policy(env, policy, device)
                
                if CONFIG["debug_level"] > 0:
                    print(f"Evaluation at step {step}:")
                    print(f"  Mean return: {eval_results['mean_return']:.2f}")
                    print(f"  Success rate: {eval_results['success_rate']:.2f}")
                
                # Save best model
                if eval_results['success_rate'] > best_success_rate:
                    best_success_rate = eval_results['success_rate']
                    best_model_path = os.path.join(
                        CONFIG["checkpoint_dir"],
                        f"{CONFIG['model_name']}_best.pt"
                    )
                    torch.save({
                        'step': step,
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'success_rate': best_success_rate,
                    }, best_model_path)
                    
                    if CONFIG["debug_level"] > 0:
                        print(f"  New best model saved with success rate: {best_success_rate:.2f}")
                
                # Update logs
                log_data['steps'].append(step)
                log_data['episodes'].append(episode)
                log_data['returns'].append(eval_results['mean_return'])
                log_data['success_rates'].append(eval_results['success_rate'])
        
        # Update policy using PPO
        update_results = ppo_update(policy, optimizer, buffer, device, CONFIG)
        
        # Log update statistics
        if CONFIG["debug_level"] > 0:
            print(f"Policy update at step {step}:")
            print(f"  Loss: {update_results['loss']:.4f}")
            print(f"  Value loss: {update_results['value_loss']:.4f}")
            print(f"  Policy loss: {update_results['policy_loss']:.4f}")
            print(f"  Entropy: {update_results['entropy']:.4f}")
        
        # Update logs
        log_data['losses'].append(update_results['loss'])
        log_data['value_losses'].append(update_results['value_loss'])
        log_data['policy_losses'].append(update_results['policy_loss'])
        log_data['entropies'].append(update_results['entropy'])
        
        # Clear buffer
        buffer.clear()
    
    # Save final model
    final_model_path = os.path.join(
        CONFIG["checkpoint_dir"],
        f"{CONFIG['model_name']}_final.pt"
    )
    torch.save({
        'step': step,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    
    print("Training completed!")
    return policy, log_data

def plot_training_progress(log_data):
    """Plot training statistics"""
    plt.figure(figsize=(15, 10))
    
    # Plot returns
    plt.subplot(2, 2, 1)
    plt.plot(log_data['steps'], log_data['returns'])
    plt.title('Mean Return')
    plt.xlabel('Steps')
    plt.ylabel('Return')
    
    # Plot success rate
    plt.subplot(2, 2, 2)
    plt.plot(log_data['steps'], log_data['success_rates'])
    plt.title('Success Rate')
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    
    # Plot losses
    plt.subplot(2, 2, 3)
    plt.plot(log_data['steps'][::CONFIG["buffer_size"]], log_data['losses'], label='Total Loss')
    plt.plot(log_data['steps'][::CONFIG["buffer_size"]], log_data['value_losses'], label='Value Loss')
    plt.plot(log_data['steps'][::CONFIG["buffer_size"]], log_data['policy_losses'], label='Policy Loss')
    plt.title('Training Losses')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot entropy
    plt.subplot(2, 2, 4)
    plt.plot(log_data['steps'][::CONFIG["buffer_size"]], log_data['entropies'])
    plt.title('Policy Entropy')
    plt.xlabel('Steps')
    plt.ylabel('Entropy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "training_progress.png"))
    plt.show()

def test_model(model_path, num_episodes=10, render=True):
    """Test a trained model"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = DroneLandingEnv()
    
    # Create policy network
    policy = ViTDronePolicy(
        image_size=CONFIG["image_size"],
        patch_size=CONFIG["patch_size"],
        lidar_points=CONFIG["lidar_points"],
        hidden_dim=CONFIG["hidden_dim"]
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Testing for {num_episodes} episodes...")
    
    # Test variables
    episode_returns = []
    success_count = 0
    flight_durations = []
    landing_velocities = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Convert observation to tensor
            camera_img = torch.FloatTensor(obs['camera']).unsqueeze(0).to(device)
            lidar_data = torch.FloatTensor(obs['lidar']).unsqueeze(0).to(device)
            
            # Get action
            with torch.no_grad():
                action, _, _ = policy.act(camera_img, lidar_data)
            
            # Take step
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Render if enabled
            if render:
                env.render()
        
        # Log episode statistics
        episode_returns.append(episode_reward)
        flight_durations.append(steps)
        
        if info.get('landed', False):
            success_count += 1
            landing_velocities.append(info['velocity'])
            print(f"Episode {episode}: SUCCESS - Reward: {episode_reward:.2f}, Steps: {steps}")
        else:
            print(f"Episode {episode}: FAILED - Reward: {episode_reward:.2f}, Steps: {steps}")
    
    # Print overall statistics
    success_rate = success_count / num_episodes
    print("\nTest Results:")
    print(f"Success Rate: {success_rate:.2f} ({success_count}/{num_episodes})")
    print(f"Mean Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean Flight Duration: {np.mean(flight_durations):.2f} steps")
    
    if success_count > 0:
        print(f"Mean Landing Velocity: {np.mean(landing_velocities):.2f} m/s")
    
    return {
        'success_rate': success_rate,
        'mean_return': np.mean(episode_returns),
        'returns': episode_returns,
        'flight_durations': flight_durations,
        'landing_velocities': landing_velocities if success_count > 0 else []
    }

import argparse
import os
import sys

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Drone Landing RL Training and Testing')
    
    # Main operation mode
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                        help='Operation mode: train or test')
    
    # Training arguments
    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='Total timesteps for training')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save/load checkpoints')
    parser.add_argument('--model-name', type=str, default='drone_landing_model',
                        help='Base name for model checkpoints')
    
    # Testing arguments
    parser.add_argument('--model-path', type=str,
                        help='Path to model checkpoint for testing')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to test')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Update configuration based on arguments
    CONFIG["render"] = args.render
    CONFIG["checkpoint_dir"] = args.checkpoint_dir
    CONFIG["model_name"] = args.model_name
    
    # Create checkpoint directory
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    
    if args.mode == 'train':
        CONFIG["total_timesteps"] = args.timesteps
        print(f"Starting training for {args.timesteps} timesteps...")
        policy, log_data = train()
        plot_training_progress(log_data)
        
        # Test the best model after training
        best_model_path = os.path.join(
            CONFIG["checkpoint_dir"],
            f"{CONFIG['model_name']}_best.pt"
        )
        print("\nTesting best model...")
        test_model(best_model_path, num_episodes=5, render=True)
        
    elif args.mode == 'test':
        # Check if model path is provided
        if not args.model_path:
            print("Error: --model-path is required when mode is 'test'")
            sys.exit(1)
            
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found: {args.model_path}")
            sys.exit(1)
            
        print(f"Testing model from {args.model_path} for {args.episodes} episodes...")
        test_model(args.model_path, num_episodes=args.episodes, render=True)