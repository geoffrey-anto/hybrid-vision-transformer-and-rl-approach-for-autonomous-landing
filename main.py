"""
Drone Landing Reinforcement Learning Project

This project trains a drone to land on even terrain using:
- PPO (Proximal Policy Optimization) algorithm
- Vision transformer for image processing
- LiDAR data integration
- Multi-core training capabilities
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import logging
import gym
from gym import spaces
import random
from concurrent.futures import ProcessPoolExecutor
from collections import deque


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("drone_landing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("drone_landing")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define the Drone Environment
class DroneEnvironment(gym.Env):
    def __init__(self, terrain_size=100, starting_height=10, render_mode=None):
        super(DroneEnvironment, self).__init__()
        
        # Environment parameters
        self.terrain_size = terrain_size
        self.starting_height = starting_height
        self.max_steps = 100
        self.current_step = 0
        self.render_mode = render_mode
        
        # Action space: 0 = left, 1 = right, 2 = down
        self.action_space = spaces.Discrete(3)
        
        # Observation space: camera (RGB) and LiDAR data
        self.camera_shape = (3, 64, 64)  # RGB image
        self.lidar_shape = (16,)  # 16 LiDAR readings
        
        self.observation_space = spaces.Dict({
            'camera': spaces.Box(low=0, high=255, shape=self.camera_shape, dtype=np.uint8),
            'lidar': spaces.Box(low=0, high=100, shape=self.lidar_shape, dtype=np.float32)
        })
        
        # Initialize terrain, drone position, and history
        self.reset()
        
        # For visualization
        self.trajectory = []

    def generate_terrain(self):
        """Generate a terrain with even and uneven parts."""
        # Base terrain
        terrain = np.zeros(self.terrain_size)
        
        # Add some uneven areas (represented by higher values)
        for _ in range(5):
            start = random.randint(0, self.terrain_size - 20)
            length = random.randint(5, 15)
            height = random.uniform(0.5, 2.0)
            terrain[start:start+length] = height
        
        # Smooth the terrain
        terrain = np.convolve(terrain, np.ones(5)/5, mode='same')
        
        return terrain

    def is_even_terrain(self, position):
        """Check if the current position is on even terrain."""
        # Check current position and a small window around it
        left_idx = max(0, int(position) - 2)
        right_idx = min(self.terrain_size - 1, int(position) + 2)
        
        # Calculate the variance of the terrain height in this region
        region = self.terrain[left_idx:right_idx+1]
        variance = np.var(region)
        
        # Low variance means even terrain
        return variance < 0.05
    
    def get_observation(self):
        """Get camera and LiDAR observations."""
        # Generate camera view
        camera_obs = np.zeros(self.camera_shape, dtype=np.uint8)
        
        # Add terrain features to the bottom of the image
        terrain_slice = self.terrain[max(0, int(self.drone_x) - 32):min(self.terrain_size, int(self.drone_x) + 32)]
        if len(terrain_slice) < 64:
            padding = 64 - len(terrain_slice)
            terrain_slice = np.pad(terrain_slice, (0, padding), 'constant')
        
        # Normalize terrain to image height
        max_height = 20
        normalized_terrain = (terrain_slice / max_height * 30).astype(int)
        
        # Add terrain to image
        for i in range(64):
            h = int(normalized_terrain[i])
            camera_obs[0, 63-h:64, i] = 100  # Red channel
            camera_obs[1, 63-h:64, i] = 100  # Green channel
            
        # Add drone to the image
        drone_y_pixel = int(63 - (self.drone_y / max_height * 30))
        drone_x_pixel = 32  # Center of the image
        
        # Draw drone as a blue dot
        for c in range(-2, 3):
            for r in range(-2, 3):
                if 0 <= drone_y_pixel + r < 64 and 0 <= drone_x_pixel + c < 64:
                    camera_obs[2, drone_y_pixel + r, drone_x_pixel + c] = 255  # Blue channel
        
        # Generate LiDAR readings
        lidar_obs = np.zeros(self.lidar_shape, dtype=np.float32)
        for i in range(16):
            # Position relative to drone
            rel_x = int(self.drone_x) - 8 + i
            if 0 <= rel_x < self.terrain_size:
                # Distance to ground
                lidar_obs[i] = self.drone_y - self.terrain[rel_x]
            else:
                lidar_obs[i] = self.drone_y  # Out of bounds, return height above ground level
        
        return {
            'camera': camera_obs,
            'lidar': lidar_obs
        }
    
    def reset(self, seed=None):
        """Reset the environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # Generate terrain
        self.terrain = self.generate_terrain()
        
        # Place drone at random x position and fixed starting height
        self.drone_x = random.uniform(0, self.terrain_size - 1)
        self.drone_y = self.starting_height
        
        # Reset step counter
        self.current_step = 0
        
        # Reset trajectory
        self.trajectory = [(self.drone_x, self.drone_y)]
        
        # Get initial observation
        observation = self.get_observation()
        
        # Return observation and info
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Take an action in the environment."""
        self.current_step += 1
        
        # Move drone based on action
        if action == 0:  # Left
            self.drone_x = max(0, self.drone_x - 1.0)
        elif action == 1:  # Right
            self.drone_x = min(self.terrain_size - 1, self.drone_x + 1.0)
        elif action == 2:  # Down
            self.drone_y = max(0, self.drone_y - 0.5)
            
        # Update trajectory
        self.trajectory.append((self.drone_x, self.drone_y))
        
        # Get height of terrain at current position
        terrain_height = self.terrain[int(self.drone_x)]
        
        # Check if landed (drone height <= terrain height)
        landed = self.drone_y <= terrain_height
        
        # Check if landing is on even terrain
        on_even_terrain = self.is_even_terrain(self.drone_x)
        
        # Default reward: small penalty for fuel consumption
        reward = -0.01
        
        done = False
        info = {
            "landed": False,
            "landing_success": False
        }
        
        if landed:
            done = True
            info["landed"] = True
            
            if on_even_terrain:
                # Good landing on even terrain
                reward = 10.0
                info["landing_success"] = True
                logger.info(f"Successful landing at position {self.drone_x:.2f} with height {terrain_height:.2f}")
            else:
                # Bad landing on uneven terrain
                reward = -5.0
                logger.info(f"Failed landing at position {self.drone_x:.2f} with height {terrain_height:.2f} (uneven terrain)")
        elif self.current_step >= self.max_steps:
            # Out of time/fuel
            done = True
            reward = -1.0
            logger.info("Episode timed out without landing")
            
        # Get new observation
        observation = self.get_observation()
        
        return observation, reward, done, False, info  # False is for truncated in gym step API
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            plt.figure(figsize=(10, 6))
            
            # Plot terrain
            x = np.arange(self.terrain_size)
            plt.plot(x, self.terrain, 'g-')
            
            # Plot drone trajectory
            traj_x, traj_y = zip(*self.trajectory)
            plt.plot(traj_x, traj_y, 'b-')
            
            # Plot current drone position
            plt.plot(self.drone_x, self.drone_y, 'ro')
            
            plt.title("Drone Landing Simulation")
            plt.xlabel("Position")
            plt.ylabel("Height")
            plt.ylim(0, self.starting_height + 1)
            
            plt.draw()
            plt.pause(0.001)
            plt.close()

# Neural Network Models
class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, embedding_dim=128, num_heads=4, num_layers=4):
        super(VisionTransformer, self).__init__()
        
        # Calculate number of patches
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels, embedding_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embedding_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output MLP
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        batch_size = x.shape[0]
        
        # Create patch embeddings
        x = self.patch_embedding(x)  # [batch_size, embedding_dim, height/patch_size, width/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, embedding_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embedding[:, :(x.size(1))]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Take class token for output
        x = x[:, 0]
        
        # Apply MLP
        x = self.mlp_head(x)
        
        return x

class LidarProcessor(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=128):
        super(LidarProcessor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class ResNet50Processor(nn.Module):
    def __init__(self, output_dim=128):
        super(ResNet50Processor, self).__init__()
        
        # We're implementing a simplified version for this example
        # In a real implementation, you would use torchvision.models.resnet50
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Simplified ResNet blocks
        self.layer1 = self._make_layer(16, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2)
        
        # Global average pooling and final layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_dim)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        # First block handles downsampling
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class SimpleCNN(nn.Module):
    def __init__(self, output_dim=128):
        super(SimpleCNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class FeatureExtractor(nn.Module):
    def __init__(self, model_type="vit", fusion_dim=256):
        super(FeatureExtractor, self).__init__()
        
        # Initialize model based on type
        if model_type == "vit":
            self.image_processor = VisionTransformer(
                img_size=64, 
                patch_size=8, 
                in_channels=3, 
                embedding_dim=128
            )
        elif model_type == "resnet50":
            self.image_processor = ResNet50Processor(output_dim=128)
        elif model_type == "cnn":
            self.image_processor = SimpleCNN(output_dim=128)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.lidar_processor = LidarProcessor(input_dim=16, output_dim=128)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        self.model_type = model_type
        
    def forward(self, observation):
        # Process image from camera
        camera_data = observation['camera'].float() / 255.0  # Normalize to [0, 1]
        image_features = self.image_processor(camera_data)
        
        # Process LiDAR data
        lidar_data = observation['lidar']
        lidar_features = self.lidar_processor(lidar_data)
        
        # Concatenate features
        combined_features = torch.cat([image_features, lidar_features], dim=1)
        
        # Fuse features
        fused_features = self.fusion(combined_features)
        
        return fused_features

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim=256, num_actions=3):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
    def forward(self, x):
        action_logits = self.network(x)
        return action_logits

class ValueNetwork(nn.Module):
    def __init__(self, feature_dim=256):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class PPOAgent:
    def __init__(self, 
                 model_type="vit",
                 gamma=0.99, 
                 gae_lambda=0.95,
                 clip_param=0.2,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 lr=3e-4,
                 batch_size=64,
                 ppo_epochs=10,
                 device=device):
        
        self.model_type = model_type
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
        self.device = device
        
        # Initialize networks
        self.feature_extractor = FeatureExtractor(model_type=model_type).to(device)
        self.policy = PolicyNetwork().to(device)
        self.value = ValueNetwork().to(device)
        
        # Setup optimizers
        self.optimizer = optim.Adam([
            {'params': self.feature_extractor.parameters()},
            {'params': self.policy.parameters()},
            {'params': self.value.parameters()}
        ], lr=lr)
        
        # For saving and loading
        self.best_reward = float('-inf')
        
    def get_action(self, observation, evaluate=False):
        # Convert observation to tensors
        camera = torch.FloatTensor(observation['camera']).unsqueeze(0).to(self.device) / 255.0
        lidar = torch.FloatTensor(observation['lidar']).unsqueeze(0).to(self.device)
        
        processed_observation = {
            'camera': camera,
            'lidar': lidar
        }
        
        # Get features
        with torch.no_grad():
            features = self.feature_extractor(processed_observation)
            action_logits = self.policy(features)
            value = self.value(features)
        
        # Get action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Sample action or get most likely action
        if evaluate:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        # Get log probability
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def update(self, rollouts):
        """Update policy and value networks using PPO algorithm."""
        # Prepare data
        observations = rollouts['observations']
        actions = torch.LongTensor(rollouts['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollouts['log_probs']).to(self.device)
        rewards = rollouts['rewards']
        dones = rollouts['dones']
        values = rollouts['values']
        
        # Compute returns and advantages
        advantages = self.compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns (value targets)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        # Prepare minibatches
        batch_size = min(self.batch_size, len(observations))
        indices = np.arange(len(observations))
        
        # PPO update loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(observations), batch_size):
                end_idx = min(start_idx + batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]
                
                # Process batch observations
                batch_camera = torch.FloatTensor(
                    np.stack([observations[i]['camera'] for i in batch_indices])
                ).to(self.device) / 255.0
                
                batch_lidar = torch.FloatTensor(
                    np.stack([observations[i]['lidar'] for i in batch_indices])
                ).to(self.device)
                
                batch_observations = {
                    'camera': batch_camera,
                    'lidar': batch_lidar
                }
                
                # Get new features, action logits, and values
                features = self.feature_extractor(batch_observations)
                action_logits = self.policy(features)
                current_values = self.value(features).squeeze(-1)
                
                # Get action distribution
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                
                # Get batch items
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute new log probs and entropy
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(current_values, batch_returns)
                
                # Compute total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Update networks
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.feature_extractor.parameters()) + 
                    list(self.policy.parameters()) + 
                    list(self.value.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Average losses over epochs and batches
        num_updates = self.ppo_epochs * ((len(observations) + batch_size - 1) // batch_size)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }
    
    def save_checkpoint(self, path, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'model_type': self.model_type,
            'feature_extractor': self.feature_extractor.state_dict(),
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_reward': self.best_reward
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved new best model with reward {self.best_reward}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check model type
        if checkpoint['model_type'] != self.model_type:
            logger.warning(f"Loading checkpoint with different model type: {checkpoint['model_type']} vs {self.model_type}")
            
            # Reinitialize feature extractor with correct model type
            self.feature_extractor = FeatureExtractor(model_type=checkpoint['model_type']).to(self.device)
            self.model_type = checkpoint['model_type']
        
        # Load network states
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load best reward
        self.best_reward = checkpoint['best_reward']
        
        logger.info(f"Loaded checkpoint with best reward {self.best_reward}")
        
        return checkpoint

def train(agent, env, num_episodes=1000, checkpoint_dir="checkpoints", log_interval=10, checkpoint_interval=50, num_workers=4):
    """Train the agent using PPO."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_dir, agent.model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create metrics tracking
    all_rewards = []
    episode_rewards = []
    success_rate = []
    
    # For visualization
    plt.figure(figsize=(12, 8))

    global_average_reward = 0.00
    global_loss = 0.00
    global_landing_success = 0.00
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        # Initialize episode data
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        # Reset environment
        observation, _ = env.reset()
        
        done = False
        episode_reward = 0
        
        # Episode loop
        while not done:
            # Get action
            action, log_prob, value = agent.get_action(observation)
            
            # Take action in environment
            next_observation, reward, done, _, info = env.step(action)
            
            # Store data
            observations.append(observation)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            
            # Update for next step
            observation = next_observation
            episode_reward += reward

        global_average_reward += episode_reward
        
        # Track episode metrics
        episode_rewards.append(episode_reward)
        if episode >= log_interval and episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            all_rewards.append(avg_reward)
        success_rate.append(1 if info.get("landing_success", False) else 0)

        global_landing_success += info.get("landing_success", False)
        
        # Prepare rollout data
        rollouts = {
            'observations': observations,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'dones': dones,
            'values': values
        }
        
        # Update agent
        loss_info = agent.update(rollouts)

        global_loss += loss_info['policy_loss'] + loss_info['value_loss'] + loss_info['entropy']
        
        # Logging
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_success = np.mean(success_rate[-log_interval:]) * 100
            
            logger.info(f"Episode {episode}/{num_episodes} | " +
                        f"Avg Reward: {avg_reward:.2f} | " +
                        f"Success Rate: {avg_success:.2f}% | " +
f"Loss: Policy={loss_info['policy_loss']:.4f}, Value={loss_info['value_loss']:.4f}, Entropy={loss_info['entropy']:.4f}")
        
        # Update visualization
        if episode % log_interval == 0:
            plt.clf()
            
            # Plot rewards
            plt.subplot(2, 2, 1)
            plt.plot(np.arange(1, len(all_rewards) + 1), all_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            # Plot success rate (moving average)
            plt.subplot(2, 2, 2)
            window_size = min(100, len(success_rate))
            moving_avg = [np.mean(success_rate[max(0, i - window_size):i + 1]) * 100 
                         for i in range(len(success_rate))]
            plt.plot(np.arange(1, len(success_rate) + 1), moving_avg)
            plt.title('Landing Success Rate (%)')
            plt.xlabel('Episode')
            plt.ylabel('Success Rate')
            plt.ylim(0, 100)
            
            # Plot the latest trajectory
            plt.subplot(2, 2, 3)
            traj_x, traj_y = zip(*env.trajectory)
            plt.plot(np.arange(len(env.terrain)), env.terrain, 'g-', label='Terrain')
            plt.plot(traj_x, traj_y, 'b-', label='Drone Path')
            plt.scatter(traj_x[-1], traj_y[-1], color='r', s=50, label='Landing Point')
            plt.title('Latest Drone Trajectory')
            plt.xlabel('Position')
            plt.ylabel('Height')
            plt.legend()
            
            # Plot policy loss
            plt.subplot(2, 2, 4)
            plt.title('Training Metrics')
            plt.xlabel('Episode (Sampled)')
            plt.ylabel('Value')
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(checkpoint_dir, 'training_progress.png'))
            plt.draw()
            plt.pause(0.001)
            
        # Save checkpoint
        if episode % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_episode_{episode}.pth')
            agent.save_checkpoint(checkpoint_path)
        
        # Check if this is the best model so far
        if np.mean(episode_rewards[-10:]) > agent.best_reward:
            agent.best_reward = np.mean(episode_rewards[-10:])
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            agent.save_checkpoint(best_model_path, is_best=True)

    # Final evaluation
    print("Training complete. Evaluating final model...")
    logger.info("Training complete. Evaluating final model...")

    print(f"Global Average Reward: {global_average_reward / num_episodes:.2f}")
    logger.info(f"Global Average Reward: {global_average_reward / num_episodes:.2f}")

    print(f"Global Loss: {global_loss / num_episodes:.4f}")
    logger.info(f"Global Loss: {global_loss / num_episodes:.4f}")

    print(f"Best Reward: {agent.best_reward:.2f}")
    logger.info(f"Best Reward: {agent.best_reward:.2f}")

    print(f"Global Landing Success Rate: {global_landing_success / num_episodes * 100:.2f}%")
    logger.info(f"Global Landing Success Rate: {global_landing_success / num_episodes * 100:.2f}%")

    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    agent.save_checkpoint(final_model_path)
    
    plt.close()
    
    return agent

def parallel_evaluate(args):
    """Function for parallel evaluation of the model."""
    agent, model_path, episode = args
    
    # Load model
    agent.load_checkpoint(model_path)
    
    # Create environment
    env = DroneEnvironment(starting_height=10)
    
    # Run evaluation episode
    observation, _ = env.reset()
    done = False
    total_reward = 0
    trajectory = []
    
    while not done:
        action, _, _ = agent.get_action(observation, evaluate=True)
        next_observation, reward, done, _, info = env.step(action)
        
        observation = next_observation
        total_reward += reward
        trajectory.append((env.drone_x, env.drone_y))
    
    # Return results
    return {
        'episode': episode,
        'reward': total_reward,
        'success': info.get('landing_success', False),
        'trajectory': trajectory,
        'terrain': env.terrain
    }

def evaluate(agent, model_path, num_episodes=100, num_workers=4):
    """Evaluate a trained model."""
    logger.info(f"Evaluating model: {model_path}")
    
    # Setup parallel workers
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        args = [(agent, model_path, i) for i in range(num_episodes)]
        results = list(executor.map(parallel_evaluate, args))
    
    # Compile results
    rewards = [r['reward'] for r in results]
    success_count = sum(1 for r in results if r['success'])
    
    # Log results
    avg_reward = np.mean(rewards)
    success_rate = success_count / num_episodes * 100
    
    logger.info(f"Evaluation Results:")
    logger.info(f"Average Reward: {avg_reward:.2f}")
    logger.info(f"Success Rate: {success_rate:.2f}%")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot reward distribution
    plt.subplot(2, 2, 1)
    plt.hist(rewards, bins=20)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    
    # Plot success rate
    plt.subplot(2, 2, 2)
    labels = ['Success', 'Failure']
    sizes = [success_count, num_episodes - success_count]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Landing Success Rate')
    
    # Plot sample trajectories (successful and failed)
    plt.subplot(2, 2, 3)
    
    # Find successful and failed examples
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    # Plot successful trajectory if available
    if successful:
        sample = random.choice(successful)
        traj_x, traj_y = zip(*sample['trajectory'])
        plt.plot(np.arange(len(sample['terrain'])), sample['terrain'], 'g-', label='Terrain')
        plt.plot(traj_x, traj_y, 'b-', label='Drone Path')
        plt.scatter(traj_x[-1], traj_y[-1], color='r', s=50, label='Landing Point')
        plt.title('Successful Landing Example')
        plt.xlabel('Position')
        plt.ylabel('Height')
        plt.legend()
    
    # Plot failed trajectory if available
    plt.subplot(2, 2, 4)
    if failed:
        sample = random.choice(failed)
        traj_x, traj_y = zip(*sample['trajectory'])
        plt.plot(np.arange(len(sample['terrain'])), sample['terrain'], 'g-', label='Terrain')
        plt.plot(traj_x, traj_y, 'b-', label='Drone Path')
        plt.scatter(traj_x[-1], traj_y[-1], color='r', s=50, label='Landing Point')
        plt.title('Failed Landing Example')
        plt.xlabel('Position')
        plt.ylabel('Height')
        plt.legend()
    
    # Save plot
    plt.tight_layout()
    evaluation_dir = os.path.dirname(model_path)
    plt.savefig(os.path.join(evaluation_dir, 'evaluation_results.png'))
    plt.show()
    
    return {
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'results': results
    }

def compare_models(model_paths, num_episodes=50, num_workers=4):
    """Compare different models (e.g., VIT vs ResNet50 vs CNN)."""
    logger.info("Comparing models:")
    for path in model_paths:
        logger.info(f"  - {path}")
    
    results = {}
    
    for path in model_paths:
        # Extract model type from path
        model_type = os.path.basename(os.path.dirname(path))
        
        # Create agent with correct model type
        agent = PPOAgent(model_type=model_type)
        
        # Evaluate model
        eval_results = evaluate(agent, path, num_episodes=num_episodes, num_workers=num_workers)
        
        # Store results
        results[model_type] = eval_results
    
    # Compare results
    plt.figure(figsize=(12, 6))
    
    # Plot average rewards
    plt.subplot(1, 2, 1)
    model_types = list(results.keys())
    avg_rewards = [results[model]['avg_reward'] for model in model_types]
    plt.bar(model_types, avg_rewards)
    plt.title('Average Reward by Model Type')
    plt.xlabel('Model Type')
    plt.ylabel('Average Reward')
    
    # Plot success rates
    plt.subplot(1, 2, 2)
    success_rates = [results[model]['success_rate'] for model in model_types]
    plt.bar(model_types, success_rates)
    plt.title('Success Rate by Model Type')
    plt.xlabel('Model Type')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    
    # Save comparison
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # Log comparison
    logger.info("\nModel Comparison Results:")
    for model in model_types:
        logger.info(f"{model}:")
        logger.info(f"  Average Reward: {results[model]['avg_reward']:.2f}")
        logger.info(f"  Success Rate: {results[model]['success_rate']:.2f}%")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Drone Landing RL with PPO")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'compare'],
                       help='Mode to run the script in')
    parser.add_argument('--model_type', type=str, default='vit', choices=['vit', 'resnet50', 'cnn'],
                       help='Type of vision model to use')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint for evaluation')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of episodes for training or evaluation')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                       help='Interval for saving checkpoints during training')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Interval for logging during training')
    parser.add_argument('--starting_height', type=int, default=10,
                       help='Starting height of the drone')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker cores to use for parallelization')
    
    args = parser.parse_args()
    
    # Setup
    checkpoint_dir = os.path.join("checkpoints", args.model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Run in specified mode
    if args.mode == 'train':
        logger.info(f"Training drone landing with model type: {args.model_type}")
        
        # Create environment and agent
        env = DroneEnvironment(starting_height=args.starting_height)
        agent = PPOAgent(model_type=args.model_type)
        
        # Train agent
        trained_agent = train(
            agent, 
            env, 
            num_episodes=args.num_episodes,
            checkpoint_dir="checkpoints",
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            num_workers=args.num_workers
        )
        
        logger.info("Training completed!")
        
    elif args.mode == 'evaluate':
        if args.model_path is None:
            args.model_path = os.path.join(checkpoint_dir, "best_model.pth")
            
        logger.info(f"Evaluating model: {args.model_path}")
        
        # Create agent
        agent = PPOAgent(model_type=args.model_type)
        
        # Evaluate agent
        evaluation_results = evaluate(
            agent, 
            args.model_path, 
            num_episodes=args.num_episodes,
            num_workers=args.num_workers
        )
        
        logger.info("Evaluation completed!")
        
    elif args.mode == 'compare':
        logger.info("Comparing different model architectures")
        
        # Find best model for each type
        model_paths = []
        for model_type in ['vit', 'resnet50', 'cnn']:
            model_path = os.path.join("checkpoints", model_type, "best_model.pth")
            if os.path.exists(model_path):
                model_paths.append(model_path)
            else:
                logger.warning(f"No model found for {model_type}")
        
        if not model_paths:
            logger.error("No models found for comparison. Train models first.")
        else:
            # Compare models
            comparison_results = compare_models(
                model_paths, 
                num_episodes=args.num_episodes,
                num_workers=args.num_workers
            )
            
            logger.info("Comparison completed!")