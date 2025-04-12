import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
import logging
from typing import Dict, List, Tuple, Type, Optional, Union
from collections import deque
import timm
import cv2
import torchvision.models as models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("drone_landing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DroneTraining")

# Constants for the environment
DRONE_HEIGHT = 20  # Start height in meters
TERRAIN_SIZE = (100, 100)  # Terrain grid size
EVEN_TERRAIN_THRESHOLD = 0.1  # Height difference threshold for even terrain
MAX_EPISODE_STEPS = 200

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

class TerrainGenerator:
    """Generates terrains with even and uneven surfaces for drone landing."""
    
    def __init__(self, size=(100, 100), complexity=0.3, noise_scale=2.0):
        self.size = size
        self.complexity = complexity
        self.noise_scale = noise_scale
        
    def generate_terrain(self):
        """Generate a terrain with some even spots and some uneven/rough areas."""
        # Generate base terrain using perlin noise
        x = np.linspace(0, self.complexity, self.size[0])
        y = np.linspace(0, self.complexity, self.size[1])
        x_grid, y_grid = np.meshgrid(x, y)
        
        # Generate primary terrain with perlin-like noise
        terrain = np.sin(x_grid * 5) * np.cos(y_grid * 5)
        terrain += np.random.normal(0, 0.1, self.size)  # Add some noise
        
        # Create flat zones (even terrain for landing)
        num_flat_zones = np.random.randint(3, 8)
        for _ in range(num_flat_zones):
            center_x = np.random.randint(0, self.size[0])
            center_y = np.random.randint(0, self.size[1])
            radius = np.random.randint(5, 15)
            
            for i in range(max(0, center_x - radius), min(self.size[0], center_x + radius)):
                for j in range(max(0, center_y - radius), min(self.size[1], center_y + radius)):
                    if ((i - center_x) ** 2 + (j - center_y) ** 2) < radius ** 2:
                        # Make this area flat
                        height_value = terrain[center_y, center_x]
                        terrain[j, i] = height_value
        
        # Normalize the terrain
        terrain = (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))
        
        return terrain
    
    def evaluate_landing_spot(self, terrain, x, y, radius=3):
        """Check if a landing spot is even (flat) enough."""
        x, y = int(x), int(y)
        if x < radius or y < radius or x >= self.size[0] - radius or y >= self.size[1] - radius:
            return False
        
        # Extract the landing area
        landing_area = terrain[y-radius:y+radius+1, x-radius:x+radius+1]
        
        # Calculate the max height difference
        height_diff = np.max(landing_area) - np.min(landing_area)
        
        return height_diff < EVEN_TERRAIN_THRESHOLD

    def get_lidar_reading(self, terrain, drone_x, drone_y, drone_height, num_points=36):
        """
        Simulate a LIDAR reading from the drone's current position.
        Returns distances in all directions.
        """
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        lidar_readings = []
        
        for angle in angles:
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Trace the ray down to the terrain
            current_x, current_y = drone_x, drone_y
            distance = 0
            
            while True:
                current_x += dx
                current_y += dy
                distance += 1
                
                # Check if out of bounds
                if (current_x < 0 or current_x >= terrain.shape[1] or 
                    current_y < 0 or current_y >= terrain.shape[0]):
                    lidar_readings.append(float('inf'))
                    break
                
                # Check if we hit the terrain
                terrain_height = terrain[int(current_y), int(current_x)]
                if drone_height - terrain_height <= distance:
                    lidar_readings.append(distance)
                    break
                
                # Limit to prevent infinite loops
                if distance > 100:
                    lidar_readings.append(float('inf'))
                    break
        
        return np.array(lidar_readings)
    
    def get_camera_view(self, terrain, drone_x, drone_y, drone_height, view_width=64, view_height=64):
        """
        Simulate a camera view from the drone's current position.
        Returns a top-down rendering of the terrain below.
        """
        camera_view = np.zeros((view_height, view_width))
        
        half_width = view_width // 2
        half_height = view_height // 2
        
        for i in range(view_height):
            for j in range(view_width):
                terrain_x = int(drone_x + (j - half_width))
                terrain_y = int(drone_y + (i - half_height))
                
                if (0 <= terrain_x < self.size[1] and 0 <= terrain_y < self.size[0]):
                    # Get terrain height and adjust for drone height
                    terrain_height = terrain[terrain_y, terrain_x]
                    # Normalize the height value to be between 0 and 1 based on drone height
                    relative_height = max(0, min(1, (drone_height - terrain_height) / drone_height))
                    camera_view[i, j] = relative_height
                else:
                    camera_view[i, j] = 0  # Outside terrain view
        
        # Add a third channel for RGB format expected by vision models (just repeating the grayscale)
        camera_view_rgb = np.stack([camera_view, camera_view, camera_view], axis=2)
        return camera_view_rgb

class DroneLandingEnv(gym.Env):
    """Custom Environment for drone landing task."""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, terrain_size=TERRAIN_SIZE, start_height=DRONE_HEIGHT, render_mode=None):
        super(DroneLandingEnv, self).__init__()
        
        # Environment parameters
        self.terrain_generator = TerrainGenerator(size=terrain_size)
        self.terrain = None
        self.start_height = start_height
        self.drone_height = None
        self.drone_x, self.drone_y = None, None
        self.steps = 0
        self.max_steps = MAX_EPISODE_STEPS
        self.landing_pad_locations = []  # Track flat areas suitable for landing
        self.trajectory = []  # For visualization
        self.render_mode = render_mode
        
        # Action space: Left, Right, Down
        self.action_space = spaces.Discrete(3)
        
        # Observation space: camera (64x64x3) and lidar (36)
        camera_space = spaces.Box(low=0, high=1, shape=(64, 64, 3), dtype=np.float32)
        lidar_space = spaces.Box(low=0, high=100, shape=(36,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'camera': camera_space,
            'lidar': lidar_space,
            'height': spaces.Box(low=0, high=self.start_height, shape=(1,), dtype=np.float32)
        })
        
        # For rendering
        self.fig = None
        self.ax = None
        
    def step(self, action):
        self.steps += 1
        old_position = (self.drone_x, self.drone_y, self.drone_height)
        
        # Execute action
        if action == 0:  # Move left
            self.drone_x = max(0, self.drone_x - 1)
        elif action == 1:  # Move right
            self.drone_x = min(self.terrain.shape[1] - 1, self.drone_x + 1)
        elif action == 2:  # Move down
            self.drone_height = max(0, self.drone_height - 1)
        
        # Record trajectory for visualization
        self.trajectory.append((self.drone_x, self.drone_y, self.drone_height))
        
        # Get current terrain height at drone position
        terrain_height = self.terrain[int(self.drone_y), int(self.drone_x)]
        height_above_ground = self.drone_height - terrain_height
        
        # Determine if landing occurred
        landed = height_above_ground <= 0
        
        # Calculate reward
        reward = 0
        terminated = False
        truncated = False
        info = {'landing_successful': False}
        
        if landed:
            # Check if landing spot is even
            if self.terrain_generator.evaluate_landing_spot(self.terrain, self.drone_x, self.drone_y):
                reward = 100  # High reward for successful landing
                terminated = True
                info['landing_successful'] = True
                logger.info(f"✅ Successful landing at position ({self.drone_x:.1f}, {self.drone_y:.1f})")
            else:
                reward = -50  # Penalty for landing on uneven surface
                terminated = True
                info['landing_successful'] = False
                logger.info(f"❌ Failed landing at position ({self.drone_x:.1f}, {self.drone_y:.1f}) - Uneven terrain")
        else:
            # Small reward for staying airborne (encourages the drone to be cautious)
            reward = 0.1
            
            # Additional guidance reward: higher when near even terrain
            if self.terrain_generator.evaluate_landing_spot(self.terrain, self.drone_x, self.drone_y):
                # If drone is above a good landing spot, give increasing reward as it gets closer
                reward += (1.0 / (height_above_ground + 1)) * 5
            
            # Penalty for moving away from even terrain
            if action != 2 and old_position[2] > 0:  # If moving horizontally and not landed
                old_is_even = self.terrain_generator.evaluate_landing_spot(
                    self.terrain, old_position[0], old_position[1])
                new_is_even = self.terrain_generator.evaluate_landing_spot(
                    self.terrain, self.drone_x, self.drone_y)
                
                if old_is_even and not new_is_even:
                    reward -= 2  # Penalty for moving away from good landing spot
        
        # Check if maximum steps reached
        if self.steps >= self.max_steps:
            truncated = True
            if not landed:
                reward -= 20  # Penalty for failing to land within step limit
                logger.info("⏱️ Episode timeout - Drone failed to land")
        
        # Generate observation
        observation = self._get_observation()
        
        # Logging
        if self.steps % 10 == 0 or terminated or truncated:
            logger.info(f"Step: {self.steps}, Position: ({self.drone_x:.1f}, {self.drone_y:.1f}, {self.drone_height:.1f}), "
                       f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        
        return observation, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate new terrain
        self.terrain = self.terrain_generator.generate_terrain()
        
        # Identify landing pads (flat areas) for visualization
        self.landing_pad_locations = []
        for x in range(self.terrain.shape[1]):
            for y in range(self.terrain.shape[0]):
                if self.terrain_generator.evaluate_landing_spot(self.terrain, x, y):
                    self.landing_pad_locations.append((x, y))
        
        # Reset drone position
        self.drone_x = np.random.randint(0, self.terrain.shape[1])
        self.drone_y = np.random.randint(0, self.terrain.shape[0])
        self.drone_height = self.start_height
        self.steps = 0
        self.trajectory = [(self.drone_x, self.drone_y, self.drone_height)]
        
        logger.info(f"Environment reset. Drone starting at ({self.drone_x}, {self.drone_y}, {self.drone_height})")
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Get LIDAR and camera observations
        lidar = self.terrain_generator.get_lidar_reading(
            self.terrain, self.drone_x, self.drone_y, self.drone_height)
        
        camera = self.terrain_generator.get_camera_view(
            self.terrain, self.drone_x, self.drone_y, self.drone_height)
        
        return {
            'camera': camera.astype(np.float32),
            'lidar': lidar.astype(np.float32),
            'height': np.array([self.drone_height], dtype=np.float32)
        }
    
    def render(self):
        if self.render_mode is None:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        self.ax.clear()
        
        # Plot the terrain as a heatmap
        terrain_img = self.ax.imshow(self.terrain, cmap='terrain', origin='lower')
        plt.colorbar(terrain_img, ax=self.ax, label='Terrain Height')
        
        # Plot landing pads
        landing_x = [pos[0] for pos in self.landing_pad_locations]
        landing_y = [pos[1] for pos in self.landing_pad_locations]
        self.ax.scatter(landing_x, landing_y, color='g', alpha=0.3, s=50, label='Landing Zones')
        
        # Plot drone trajectory
        if len(self.trajectory) > 1:
            traj_x = [pos[0] for pos in self.trajectory]
            traj_y = [pos[1] for pos in self.trajectory]
            self.ax.plot(traj_x, traj_y, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        
        # Plot current drone position
        self.ax.scatter(self.drone_x, self.drone_y, color='red', s=100, marker='X', label='Drone')
        
        self.ax.set_title(f'Drone Landing Simulation - Height: {self.drone_height:.1f}m')
        self.ax.set_xlabel('X position')
        self.ax.set_ylabel('Y position')
        self.ax.legend()
        
        if self.render_mode == 'human':
            plt.pause(0.1)
            return None
        elif self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

# Custom feature extractor to combine camera and LIDAR inputs
class DroneFeaturesExtractor(BaseFeaturesExtractor):
    """
    Base class for feature extractors that can be swapped for different vision models
    """
    def __init__(self, observation_space, features_dim=256):
        super(DroneFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # Extract relevant dimensions
        self.camera_shape = observation_space.spaces['camera'].shape
        self.lidar_dim = observation_space.spaces['lidar'].shape[0]
        self.height_dim = observation_space.spaces['height'].shape[0]
        
        # Vision transformer for camera input (initialize in subclasses)
        self.vision_model = None
        
        # MLP for LIDAR and height
        self.sensor_net = nn.Sequential(
            nn.Linear(self.lidar_dim + self.height_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.get_vision_output_dim() + 64, features_dim),
            nn.ReLU()
        )
    
    def get_vision_output_dim(self):
        """Must be implemented by subclasses to return vision model output dimension"""
        return 128
    
    def forward(self, observations):
        # Process camera input
        camera = observations['camera'].float() / 255.0  # Normalize
        camera_features = self.vision_model(camera)
        
        # Process LIDAR and height input
        lidar = observations['lidar'].float()
        height = observations['height'].float()
        sensor_input = torch.cat([lidar, height], dim=1)
        sensor_features = self.sensor_net(sensor_input)
        
        # Fusion
        combined_features = torch.cat([camera_features, sensor_features], dim=1)
        return self.fusion(combined_features)

class ViTFeaturesExtractor(DroneFeaturesExtractor):
    """Vision Transformer feature extractor for the drone landing task"""
    
    def __init__(self, observation_space, features_dim=256):
        super(ViTFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # Initialize ViT model from timm
        self.vision_model = timm.create_model(
            'vit_tiny_patch16_224', 
            pretrained=False, 
            num_classes=128,
            img_size=64
        )
        
        # Modify the model to accept the correct input channels (3)
        if self.camera_shape[2] != 3:
            self.vision_model.patch_embed.proj = nn.Conv2d(
                self.camera_shape[2], 
                self.vision_model.patch_embed.proj.out_channels,
                kernel_size=self.vision_model.patch_embed.proj.kernel_size,
                stride=self.vision_model.patch_embed.proj.stride,
                padding=self.vision_model.patch_embed.proj.padding
            )
        
        # Initialize all weights properly to avoid numerical instability
        for m in self.vision_model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Add batch normalization for stability
        self.bn = nn.BatchNorm1d(128)
    
    def forward(self, observations):
        try:
            # Process camera input
            camera = observations['camera'].float()
            # Scale to [0, 1] and ensure no division by zero
            max_val = torch.max(camera)
            if max_val > 0:
                camera = camera / max_val
            
            # Ensure correct dimensions [B, C, H, W]
            if camera.dim() == 3:
                camera = camera.unsqueeze(0)
            
            # Handle NaN values
            camera = torch.nan_to_num(camera, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Permute if necessary (from [B, H, W, C] to [B, C, H, W])
            if camera.shape[1] != 3 and camera.shape[-1] == 3:
                camera = camera.permute(0, 3, 1, 2)
            
            # Process through vision model
            with torch.no_grad():  # Use no_grad to prevent gradient issues during initial runs
                camera_features = self.vision_model(camera)
                camera_features = self.bn(camera_features)
                
            # Handle potential NaNs again after model processing
            camera_features = torch.nan_to_num(camera_features, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Process LIDAR and height input
            lidar = observations['lidar'].float()
            height = observations['height'].float()
            
            # Handle NaN and infinite values in sensor inputs
            lidar = torch.nan_to_num(lidar, nan=0.0, posinf=100.0, neginf=0.0)
            height = torch.nan_to_num(height, nan=0.0, posinf=20.0, neginf=0.0)
            
            sensor_input = torch.cat([lidar, height], dim=1)
            sensor_features = self.sensor_net(sensor_input)
            
            # Handle potential NaNs in sensor features
            sensor_features = torch.nan_to_num(sensor_features, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Fusion
            combined_features = torch.cat([camera_features, sensor_features], dim=1)
            
            # Final check for NaN values
            combined_features = torch.nan_to_num(combined_features, nan=0.0, posinf=1.0, neginf=0.0)
            
            return self.fusion(combined_features)
        except Exception as e:
            logger.error(f"Error in ViTFeaturesExtractor.forward: {e}")
            # Return zeros as fallback to avoid crashing
            return torch.zeros(1, self.features_dim, device=observations['camera'].device)

class ResNet50FeaturesExtractor(DroneFeaturesExtractor):
    """ResNet50 feature extractor for the drone landing task"""
    
    def __init__(self, observation_space, features_dim=256):
        super(ResNet50FeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # Initialize ResNet50 model
        resnet = models.resnet50(weights=None)
        
        # Modify first conv layer to accept potentially different input channels
        if self.camera_shape[2] != 3:
            resnet.conv1 = nn.Conv2d(
                self.camera_shape[2], 64, 
                kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Remove classification head and use features
        modules = list(resnet.children())[:-1]  # Remove the final FC layer
        self.vision_model = nn.Sequential(*modules)
        
        # Add a final layer to get desired output dimension
        self.vision_fc = nn.Linear(2048, 128)  # ResNet50 outputs 2048 features
    
    def get_vision_output_dim(self):
        return 128
    
    def forward(self, observations):
        # Process camera input
        camera = observations['camera'].float() / 255.0  # Normalize
        
        # ResNet expects a different format than ViT
        camera_features = self.vision_model(camera)
        camera_features = torch.flatten(camera_features, 1)
        camera_features = self.vision_fc(camera_features)
        
        # Process LIDAR and height input
        lidar = observations['lidar'].float()
        height = observations['height'].float()
        sensor_input = torch.cat([lidar, height], dim=1)
        sensor_features = self.sensor_net(sensor_input)
        
        # Fusion
        combined_features = torch.cat([camera_features, sensor_features], dim=1)
        return self.fusion(combined_features)

# Custom CNN for comparison
class CNNFeaturesExtractor(DroneFeaturesExtractor):
    """Simple CNN feature extractor for comparison"""
    
    def __init__(self, observation_space, features_dim=256):
        super(CNNFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # Simple CNN
        self.vision_model = nn.Sequential(
            nn.Conv2d(self.camera_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU()
        )
    
    def get_vision_output_dim(self):
        return 128

# Model Factory for different feature extractors
class ModelFactory:
    """Factory class to create different model architectures"""
    
    @staticmethod
    def create_feature_extractor(model_type, observation_space, features_dim=256):
        """Create a feature extractor based on model type"""
        if model_type.lower() == 'vit':
            return ViTFeaturesExtractor(observation_space, features_dim)
        elif model_type.lower() == 'resnet50':
            return ResNet50FeaturesExtractor(observation_space, features_dim)
        elif model_type.lower() == 'cnn':
            return CNNFeaturesExtractor(observation_space, features_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_policy_kwargs(model_type, observation_space, features_dim=256):
        """Create policy kwargs for stable-baselines3 models"""
        # Create a closure that captures features_dim
        def features_extractor_class(obs_space):
            return ModelFactory.create_feature_extractor(model_type, obs_space, features_dim)
        
        return {
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": {}  # Features dim is handled in the closure
        }

# Custom callbacks for training
class TensorboardCallback(BaseCallback):
    """Custom callback for plotting additional values in tensorboard."""
    
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if 'landing_successful' in info:
            self.logger.record('landing/success', float(info['landing_successful']))
        return True

# Function to make compatible environment
def make_env(render_mode=None):
    """Create an environment using gymnasium interface expected by stable-baselines3"""
    def _init():
        env = DroneLandingEnv(render_mode=render_mode)
        return env
    return _init

class TrainingManager:
    """Manages the training and evaluation of drone landing models"""
    
    def __init__(self, env_creator, model_type='vit', features_dim=256, 
                 log_dir='logs', checkpoint_dir='checkpoints'):
        self.env_creator = env_creator
        self.model_type = model_type
        self.features_dim = features_dim
        self.log_dir = log_dir
        self.checkpoint_dir = os.path.join(checkpoint_dir, model_type)
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup environment
        self.env = self._make_env()
        self.eval_env = self._make_env()
        
        # Setup model
        policy_kwargs = ModelFactory.create_policy_kwargs(
            model_type, self.env.observation_space, features_dim)
        
        self.model = PPO(
            "MultiInputPolicy", 
            self.env, 
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=self.log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            max_grad_norm=0.5,  # Add gradient clipping for stability
        )
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
    
    def _make_env(self, render_mode=None):
        """Create a vectorized environment"""
        def _init():
            env = DroneLandingEnv(render_mode=render_mode)
            return env
        
        env = DummyVecEnv([_init])
        env = VecMonitor(env)
        return env
    
    def _setup_callbacks(self):
        """Setup training callbacks"""
        # Checkpoint every 10000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.checkpoint_dir,
            name_prefix=f"drone_landing_{self.model_type}"
        )
        
        # Eval callback to save best model
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.checkpoint_dir,
            log_path=self.checkpoint_dir,
            eval_freq=5000,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
        
        # Custom TensorBoard callback
        tb_callback = TensorboardCallback()
        
        return [checkpoint_callback, eval_callback, tb_callback]
    
    def train(self, total_timesteps=500000):
        """Train the model"""
        logger.info(f"Starting training with model type: {self.model_type}")
        start_time = time.time()
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            tb_log_name=f"PPO_{self.model_type}"
        )
        
        # Save final model
        final_model_path = os.path.join(self.checkpoint_dir, f"final_model_{self.model_type}")
        self.model.save(final_model_path)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final model saved to {final_model_path}")
        
        return final_model_path
    
    def load_model(self, model_path):
        """Load a trained model"""
        logger.info(f"Loading model from {model_path}")
        return PPO.load(model_path, env=self.env)
    
    def evaluate(self, model_path, num_episodes=10, render=True, save_video=False):
        """Evaluate a trained model"""
        model = self.load_model(model_path)
        
        # Create a new environment for evaluation
        render_mode = 'rgb_array' if save_video or render else None
        eval_env = self._make_env(render_mode=render_mode).envs[0]
        
        # For video recording
        if save_video:
            video_dir = os.path.join(self.checkpoint_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{self.model_type}_eval_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, 10.0, (1200, 800))
        
        successes = 0
        rewards = []
        step_counts = []
        
        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            steps = 0
            
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                steps += 1
                
                if render or save_video:
                    img = eval_env.render()
                    if save_video and img is not None:
                        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                if terminated or truncated:
                    if info.get('landing_successful', False):
                        successes += 1
                        logger.info(f"Episode {episode+1}: SUCCESS! Reward: {episode_reward:.2f}, Steps: {steps}")
                    else:
                        logger.info(f"Episode {episode+1}: Failed landing. Reward: {episode_reward:.2f}, Steps: {steps}")
            
            rewards.append(episode_reward)
            step_counts.append(steps)
        
        # Close video writer if used
        if save_video:
            video.release()
            logger.info(f"Evaluation video saved to {video_path}")
        
        # Summary statistics
        success_rate = successes / num_episodes
        avg_reward = sum(rewards) / num_episodes
        avg_steps = sum(step_counts) / num_episodes
        
        logger.info(f"Evaluation Summary:")
        logger.info(f"Success Rate: {success_rate:.2f} ({successes}/{num_episodes})")
        logger.info(f"Average Reward: {avg_reward:.2f}")
        logger.info(f"Average Steps: {avg_steps:.2f}")
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'rewards': rewards,
            'step_counts': step_counts
        }
    
    def visualize_landing(self, model_path, num_episodes=1, save_plot=True):
        """Visualize the drone landing trajectory"""
        model = self.load_model(model_path)
        
        for episode in range(num_episodes):
            # Create a new environment with human rendering
            vis_env = self._make_env(render_mode='human').envs[0]
            obs, _ = vis_env.reset()
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = vis_env.step(action)
                
                # Short pause to make visualization visible
                plt.pause(0.1)
            
            if save_plot:
                # Create a final visualization with trajectory
                plt.figure(figsize=(12, 10))
                
                # Plot terrain
                plt.imshow(vis_env.terrain, cmap='terrain', origin='lower')
                plt.colorbar(label='Terrain Height')
                
                # Plot trajectory
                traj_x = [pos[0] for pos in vis_env.trajectory]
                traj_y = [pos[1] for pos in vis_env.trajectory]
                traj_z = [pos[2] for pos in vis_env.trajectory]
                
                # Color map the trajectory based on height
                plt.scatter(traj_x, traj_y, c=traj_z, cmap='viridis', 
                           s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
                plt.colorbar(label='Drone Height')
                
                # Connect trajectory points
                plt.plot(traj_x, traj_y, 'k-', alpha=0.3, linewidth=1)
                
                # Mark start and end
                plt.scatter(traj_x[0], traj_y[0], s=200, marker='o', color='blue', label='Start')
                plt.scatter(traj_x[-1], traj_y[-1], s=200, marker='X', color='red', label='Landing')
                
                # Mark landing pads
                landing_x = [pos[0] for pos in vis_env.landing_pad_locations]
                landing_y = [pos[1] for pos in vis_env.landing_pad_locations]
                plt.scatter(landing_x, landing_y, color='green', alpha=0.3, s=50, label='Landing Zones')
                
                plt.title(f'Drone Landing Trajectory (Success: {info.get("landing_successful", False)})')
                plt.xlabel('X position')
                plt.ylabel('Y position')
                plt.legend()
                
                # Save the plot
                plot_dir = os.path.join(self.checkpoint_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, f"{self.model_type}_trajectory_{int(time.time())}.png")
                plt.savefig(plot_path)
                plt.close()
                
                logger.info(f"Trajectory plot saved to {plot_path}")
                
                # 3D visualization of trajectory
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Prepare X, Y coordinates for terrain surface
                x = np.arange(0, vis_env.terrain.shape[1])
                y = np.arange(0, vis_env.terrain.shape[0])
                X, Y = np.meshgrid(x, y)
                
                # Plot terrain surface
                surf = ax.plot_surface(X, Y, vis_env.terrain, cmap='terrain', alpha=0.6)
                
                # Plot trajectory in 3D
                ax.plot3D(traj_x, traj_y, traj_z, 'blue', linewidth=2, marker='o', markersize=2)
                
                # Mark start and end
                ax.scatter(traj_x[0], traj_y[0], traj_z[0], s=100, marker='o', color='blue', label='Start')
                ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], s=100, marker='X', color='red', label='Landing')
                
                # Add color bar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Terrain Height')
                
                ax.set_title('3D Drone Landing Trajectory')
                ax.set_xlabel('X position')
                ax.set_ylabel('Y position')
                ax.set_zlabel('Height')
                ax.legend()
                
                # Save the 3D plot
                plot_3d_path = os.path.join(plot_dir, f"{self.model_type}_trajectory_3d_{int(time.time())}.png")
                plt.savefig(plot_3d_path)
                plt.close()
                
                logger.info(f"3D trajectory plot saved to {plot_3d_path}")
            
            # Close the environment
            vis_env.close()

def compare_models(model_types=['vit', 'resnet50', 'cnn'], total_timesteps=200000, eval_episodes=10):
    """Train and evaluate multiple model types for comparison"""
    results = {}
    
    for model_type in model_types:
        logger.info(f"===== Training and evaluating {model_type} model =====")
        
        # Create training manager
        trainer = TrainingManager(
            env_creator=lambda: DroneLandingEnv(),
            model_type=model_type
        )
        
        # Train the model
        model_path = trainer.train(total_timesteps=total_timesteps)
        
        # Evaluate the model
        eval_results = trainer.evaluate(
            model_path=model_path,
            num_episodes=eval_episodes,
            render=False,
            save_video=True
        )
        
        # Visualize example landing
        trainer.visualize_landing(model_path, num_episodes=1)
        
        # Store results
        results[model_type] = eval_results
    
    # Compare results
    logger.info("\n===== Model Comparison =====")
    for model_type, res in results.items():
        logger.info(f"{model_type.upper()}:")
        logger.info(f"  Success Rate: {res['success_rate']:.2f}")
        logger.info(f"  Average Reward: {res['avg_reward']:.2f}")
        logger.info(f"  Average Steps: {res['avg_steps']:.2f}")
    
    # Plot comparison results
    plt.figure(figsize=(15, 10))
    
    # Success rate comparison
    plt.subplot(2, 2, 1)
    success_rates = [results[m]['success_rate'] for m in model_types]
    plt.bar(model_types, success_rates, color='green')
    plt.title('Success Rate Comparison')
    plt.ylim(0, 1)
    plt.ylabel('Success Rate')
    
    # Average reward comparison
    plt.subplot(2, 2, 2)
    avg_rewards = [results[m]['avg_reward'] for m in model_types]
    plt.bar(model_types, avg_rewards, color='blue')
    plt.title('Average Reward Comparison')
    plt.ylabel('Average Reward')
    
    # Average steps comparison
    plt.subplot(2, 2, 3)
    avg_steps = [results[m]['avg_steps'] for m in model_types]
    plt.bar(model_types, avg_steps, color='orange')
    plt.title('Average Steps Comparison')
    plt.ylabel('Average Steps')
    
    # Reward distribution
    plt.subplot(2, 2, 4)
    for model_type in model_types:
        plt.hist(results[model_type]['rewards'], alpha=0.5, label=model_type)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    
    # Save comparison plot
    os.makedirs('comparison_results', exist_ok=True)
    comparison_plot_path = os.path.join('comparison_results', f'model_comparison_{int(time.time())}.png')
    plt.savefig(comparison_plot_path)
    plt.close()
    
    logger.info(f"Comparison plot saved to {comparison_plot_path}")
    
    return results

def main():
    """Main function to train or evaluate models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Drone Landing with RL and Vision Transformers')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'compare'],
                        help='Mode: train, eval, or compare')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'resnet50', 'cnn'],
                        help='Model type to use')
    parser.add_argument('--timesteps', type=int, default=500000, 
                        help='Total timesteps for training')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for evaluation')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--save_video', action='store_true',
                        help='Save a video of the evaluation episodes')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    if args.mode == 'train':
        trainer = TrainingManager(
            env_creator=lambda: DroneLandingEnv(),
            model_type=args.model
        )
        model_path = trainer.train(total_timesteps=args.timesteps)
        
        # Quick evaluation after training
        trainer.evaluate(
            model_path=model_path,
            num_episodes=3,
            render=args.render,
            save_video=args.save_video
        )
        
        # Visualize landing
        trainer.visualize_landing(model_path)
        
    elif args.mode == 'eval':
        if args.model_path is None:
            raise ValueError("Please provide a model path for evaluation using --model_path")
        
        trainer = TrainingManager(
            env_creator=lambda: DroneLandingEnv(),
            model_type=args.model
        )
        
        # Evaluate the model
        trainer.evaluate(
            model_path=args.model_path,
            num_episodes=args.eval_episodes,
            render=args.render,
            save_video=args.save_video
        )
        
        # Visualize landing
        trainer.visualize_landing(args.model_path)
        
    elif args.mode == 'compare':
        # Compare all model types
        compare_models(
            model_types=['vit', 'resnet50', 'cnn'],
            total_timesteps=args.timesteps,
            eval_episodes=args.eval_episodes
        )

if __name__ == "__main__":
    main()
