# drone_landing_rl.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import gym
from gym import spaces
import matplotlib.pyplot as plt
from PIL import Image
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
CONFIG = {
    'drone_height': 50.0,  # Starting height of the drone
    'terrain_size': 100,   # Size of the terrain grid
    'lidar_points': 16,    # Number of lidar scan points
    'camera_resolution': (64, 64),  # Camera resolution
    'max_steps': 200,      # Maximum steps per episode
    'learning_rate': 3e-4,
    'gamma': 0.99,         # Discount factor
    'ppo_epochs': 4,       # PPO update epochs
    'ppo_clip': 0.2,       # PPO clipping parameter
    'value_coef': 0.5,     # Value loss coefficient
    'entropy_coef': 0.01,  # Entropy coefficient
    'gae_lambda': 0.95,    # GAE lambda parameter
    'batch_size': 64,
    'buffer_size': 2048,
    'checkpoint_interval': 100,  # Save checkpoint every N episodes
    'checkpoint_dir': 'checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Create checkpoint directories
def create_checkpoint_dirs(model_name):
    base_dir = os.path.join(CONFIG['checkpoint_dir'], model_name)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

# Custom environment for drone landing
class DroneLandingEnv(gym.Env):
    def __init__(self):
        super(DroneLandingEnv, self).__init__()
        
        # Action space: 0 = Left, 1 = Right, 2 = Down
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Combined LIDAR and camera data
        lidar_shape = (CONFIG['lidar_points'],)
        camera_shape = CONFIG['camera_resolution'] + (3,)  # RGB image
        
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=0, high=np.inf, shape=lidar_shape),
            'camera': spaces.Box(low=0, high=255, shape=camera_shape, dtype=np.uint8)
        })
        
        # Initialize terrain, drone position
        self.terrain = None
        self.position = None
        self.height = None
        self.steps = 0
        
        # Generate new environment
        self.reset()
    
    def _generate_terrain(self):
        # Create terrain with mix of even and uneven surfaces
        size = CONFIG['terrain_size']
        terrain = np.zeros(size)
        
        # Create some flat areas (suitable for landing)
        flat_areas = []
        num_flat_areas = random.randint(2, 5)
        
        for _ in range(num_flat_areas):
            start = random.randint(0, size - 15)
            length = random.randint(10, 15)
            end = min(start + length, size)
            height = random.uniform(0, 5)
            terrain[start:end] = height
            flat_areas.append((start, end, height))
        
        # Fill the rest with irregular terrain
        for i in range(size):
            if terrain[i] == 0:  # Not part of a flat area
                if i > 0:
                    # Generate terrain with some continuity
                    terrain[i] = terrain[i-1] + random.uniform(-1.0, 1.0)
                else:
                    terrain[i] = random.uniform(0, 10)
        
        return terrain, flat_areas
    
    def reset(self):
        # Generate new terrain
        self.terrain, self.flat_areas = self._generate_terrain()
        
        # Initial drone position
        self.position = random.randint(5, CONFIG['terrain_size'] - 6)
        self.height = CONFIG['drone_height']
        self.steps = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        # Simulate LIDAR scans
        lidar_data = self._get_lidar_data()
        
        # Simulate camera image
        camera_data = self._get_camera_image()
        
        return {
            'lidar': lidar_data,
            'camera': camera_data
        }
    
    def _get_lidar_data(self):
        # Generate LIDAR readings at different angles
        lidar_points = CONFIG['lidar_points']
        lidar_data = np.zeros(lidar_points)
        
        # Calculate distances to terrain in different directions
        half_width = lidar_points // 2
        for i in range(lidar_points):
            # Calculate offset from current position
            offset = i - half_width
            scan_position = self.position + offset
            
            # Ensure within terrain bounds
            if 0 <= scan_position < len(self.terrain):
                # Distance from drone to terrain at this position
                lidar_data[i] = self.height - self.terrain[scan_position]
            else:
                # Out of bounds - set to max range
                lidar_data[i] = self.height
        
        return lidar_data
    
    def _get_camera_image(self):
        # Generate simulated camera image centered on drone position
        height, width = CONFIG['camera_resolution']
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw terrain
        terrain_start = max(0, self.position - width // 2)
        terrain_end = min(len(self.terrain), terrain_start + width)
        
        # Normalize terrain heights for visualization
        max_height = max(20, np.max(self.terrain) + 10)  # Ensure some headroom
        
        for i, terrain_idx in enumerate(range(terrain_start, terrain_end)):
            if terrain_idx < len(self.terrain):
                # Calculate terrain height in image coordinates
                terrain_height = int((self.terrain[terrain_idx] / max_height) * height)
                
                # Draw terrain column
                for h in range(height - terrain_height, height):
                    # Check if this is a flat area for visualization
                    is_flat = any(start <= terrain_idx < end for start, end, _ in self.flat_areas)
                    
                    if is_flat:
                        # Green for flat areas
                        image[h, i] = [0, 255, 0]
                    else:
                        # Brown for uneven terrain
                        image[h, i] = [165, 42, 42]
        
        # Draw drone
        drone_x = min(width - 1, max(0, width // 2))
        drone_y = min(height - 1, max(0, int((self.height / max_height) * height)))
        image[drone_y, drone_x] = [255, 0, 0]  # Red dot for drone
        
        return image
    
    def step(self, action):
        self.steps += 1
        
        # Execute action
        if action == 0:  # Left
            self.position = max(0, self.position - 1)
        elif action == 1:  # Right
            self.position = min(len(self.terrain) - 1, self.position + 1)
        elif action == 2:  # Down
            self.height = max(self.terrain[self.position], self.height - 1.0)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        # Get new observation
        observation = self._get_observation()
        
        info = {'position': self.position, 'height': self.height}
        
        return observation, reward, done, info
    
    def _calculate_reward(self):
        # Distance to ground
        ground_height = self.terrain[self.position]
        distance_to_ground = self.height - ground_height
        
        # Check if landed
        if distance_to_ground < 0.1:
            # Check if landed on a flat area
            on_flat_area = any(start <= self.position < end for start, end, _ in self.flat_areas)
            
            if on_flat_area:
                # Successfully landed on flat area - high reward
                return 100.0
            else:
                # Landed on irregular terrain - negative reward
                return -50.0
        
        # Small penalty for each step to encourage faster landing
        step_penalty = -0.1
        
        # Small reward for staying in air (smaller than landing penalty to encourage landing)
        stay_alive_reward = 0.05
        
        return step_penalty + stay_alive_reward
    
    def _is_done(self):
        # Done if landed or max steps reached
        ground_height = self.terrain[self.position]
        distance_to_ground = self.height - ground_height
        
        landed = distance_to_ground < 0.1
        timeout = self.steps >= CONFIG['max_steps']
        
        return landed or timeout
    
    def render(self, mode='human'):
        # Initialize the figure on first render
        if not hasattr(self, 'fig'):
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 4))
            self.terrain_line, = self.ax.plot(self.terrain, 'k-')
            self.drone_point, = self.ax.plot(self.position, self.height, 'ro')
            self.flat_areas_lines = []
            # Initialize flat area lines
            for start, end, height in self.flat_areas:
                line, = self.ax.plot(range(start, end), [height] * (end - start), 'g-', linewidth=3)
                self.flat_areas_lines.append(line)
            self.ax.set_xlim(0, len(self.terrain))
            self.ax.set_ylim(0, CONFIG['drone_height'] + 5)
        
        # Update data without recreating the plot
        self.drone_point.set_data(self.position, self.height)
        self.ax.set_title(f'Drone Position: ({self.position}, {self.height:.1f})')
        
        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Small pause to control rendering speed
        plt.pause(0.01)

# Model Architectures
class BaseModel(nn.Module):
    """Base model class that all models should extend"""
    def __init__(self, input_dims, action_dim):
        super(BaseModel, self).__init__()
        self.model_name = "base_model"
    
    def forward(self, x):
        raise NotImplementedError
    
    def get_name(self):
        return self.model_name

class VisionTransformerModel(BaseModel):
    def __init__(self, input_dims, action_dim):
        super(VisionTransformerModel, self).__init__(input_dims, action_dim)
        self.model_name = "vision_transformer"
        
        # Dimensions
        lidar_dim = input_dims['lidar']
        camera_h, camera_w, camera_c = input_dims['camera']
        
        # Lidar encoder
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # CNN for initial image feature extraction
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(camera_c, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate CNN output size
        cnn_output_h = camera_h // 8
        cnn_output_w = camera_w // 8
        cnn_output_c = 64
        
        # Flatten CNN output for transformer
        self.cnn_output_size = cnn_output_h * cnn_output_w * cnn_output_c
        
        # Simple Vision Transformer (ViT) components
        self.patch_size = cnn_output_h * cnn_output_w
        self.embed_dim = 64
        
        # Linear projection for patches
        self.linear_proj = nn.Linear(cnn_output_c, self.embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_size, self.embed_dim))
        
        # Transformer encoder layers (simplified)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Fusion of lidar and vision features
        self.fusion = nn.Sequential(
            nn.Linear(128 + self.embed_dim, 128),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # Process LIDAR data
        lidar = x['lidar'].float()
        lidar_features = self.lidar_encoder(lidar)
        
        # Process camera image
        # Convert from (B, H, W, C) to (B, C, H, W) format
        camera = x['camera'].permute(0, 3, 1, 2).float() / 255.0
        batch_size = camera.shape[0]
        
        # CNN feature extraction
        cnn_features = self.cnn_encoder(camera)
        
        # Reshape for transformer: (B, C, H, W) -> (B, H*W, C)
        cnn_features = cnn_features.reshape(batch_size, -1, cnn_features.shape[1])
        
        # Linear projection of patches
        patch_embeddings = self.linear_proj(cnn_features)
        
        # Add positional embeddings
        patch_embeddings = patch_embeddings + self.pos_embed
        
        # Transformer encoder
        transformer_output = self.transformer_encoder(patch_embeddings)
        
        # Global average pooling over patches
        vision_features = transformer_output.mean(dim=1)
        
        # Fusion of lidar and vision features
        combined_features = self.fusion(torch.cat([lidar_features, vision_features], dim=1))
        
        # Policy and value outputs
        policy_logits = self.policy(combined_features)
        value = self.value(combined_features)
        
        return policy_logits, value

class ResNet50Model(BaseModel):
    def __init__(self, input_dims, action_dim):
        super(ResNet50Model, self).__init__(input_dims, action_dim)
        self.model_name = "resnet50"
        
        # Dimensions
        lidar_dim = input_dims['lidar']
        camera_h, camera_w, camera_c = input_dims['camera']
        
        # Lidar encoder (same as ViT model)
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Simple ResNet-like blocks for demonstration
        # (In a real implementation, you would import and use torchvision.models.resnet50)
        self.conv1 = nn.Conv2d(camera_c, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Simple residual blocks
        self.res_block1 = self._make_res_block(16, 32)
        self.res_block2 = self._make_res_block(32, 64)
        
        # Calculate CNN output size
        cnn_output_h = camera_h // 8  # After downsampling
        cnn_output_w = camera_w // 8
        cnn_output_c = 64
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fusion of lidar and vision features
        self.fusion = nn.Sequential(
            nn.Linear(128 + cnn_output_c, 128),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            
            # Second convolutional layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            
            # Shortcut connection
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            ),
            
            nn.ReLU()
        )
    
    def forward(self, x):
        # Process LIDAR data
        lidar = x['lidar'].float()
        lidar_features = self.lidar_encoder(lidar)
        
        # Process camera image
        # Convert from (B, H, W, C) to (B, C, H, W) format
        camera = x['camera'].permute(0, 3, 1, 2).float() / 255.0
        
        # ResNet feature extraction
        x = self.conv1(camera)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Global average pooling
        x = self.avgpool(x)
        vision_features = x.view(x.size(0), -1)
        
        # Fusion of lidar and vision features
        combined_features = self.fusion(torch.cat([lidar_features, vision_features], dim=1))
        
        # Policy and value outputs
        policy_logits = self.policy(combined_features)
        value = self.value(combined_features)
        
        return policy_logits, value

class SimpleCNNModel(BaseModel):
    def __init__(self, input_dims, action_dim):
        super(SimpleCNNModel, self).__init__(input_dims, action_dim)
        self.model_name = "simple_cnn"
        
        # Dimensions
        lidar_dim = input_dims['lidar']
        camera_h, camera_w, camera_c = input_dims['camera']
        
        # Lidar encoder
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Simplified CNN for faster processing
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(camera_c, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Fusion of lidar and vision features
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Process LIDAR data
        lidar = x['lidar'].float()
        lidar_features = self.lidar_encoder(lidar)
        
        # Process camera image
        # Convert from (B, H, W, C) to (B, C, H, W) format
        camera = x['camera'].permute(0, 3, 1, 2).float() / 255.0
        
        # CNN feature extraction
        cnn_features = self.cnn_encoder(camera)
        vision_features = cnn_features.view(cnn_features.size(0), -1)
        
        # Fusion of lidar and vision features
        combined_features = self.fusion(torch.cat([lidar_features, vision_features], dim=1))
        
        # Policy and value outputs
        policy_logits = self.policy(combined_features)
        value = self.value(combined_features)
        
        return policy_logits, value

# PPO Agent
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size
    
    def store(self, state, action, prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches
    
    def get_all(self):
        return (
            self.states,
            self.actions,
            self.probs,
            self.values,
            self.rewards,
            self.dones
        )

class PPOAgent:
    def __init__(self, model, env, device='cpu'):
        self.model = model
        self.env = env
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Memory
        self.memory = PPOMemory(CONFIG['batch_size'])
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG['learning_rate'])
        
        # Training metrics
        self.best_reward = -float('inf')
        self.running_reward = 0
        self.episode_reward = 0
    
    def select_action(self, state):
        # Convert state to proper format for model
        states = {}
        for key, value in state.items():
            states[key] = torch.tensor([value], dtype=torch.float32).to(self.device)
        
        # Forward pass through the model
        with torch.no_grad():
            policy_logits, value = self.model(states)
            dist = Categorical(logits=policy_logits)
            action = dist.sample()
            action_prob = dist.log_prob(action)
        
        return action.item(), action_prob.item(), value.item()
    
    def learn(self):
        states, actions, old_log_probs, old_values, rewards, dones = self.memory.get_all()
        
        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(old_values, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # Calculate advantages using Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Calculate returns and advantages
        returns = torch.zeros_like(rewards)
        for t in range(len(rewards) - 1, -1, -1):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else old_values[t]
            else:
                next_value = old_values[t + 1]
            
            delta = rewards[t] + CONFIG['gamma'] * next_value * (1 - int(dones[t])) - old_values[t]
            gae = delta + CONFIG['gamma'] * CONFIG['gae_lambda'] * (1 - int(dones[t])) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + old_values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(CONFIG['ppo_epochs']):
            # Generate random mini-batches
            batches = self.memory.generate_batches()
            
            for batch_indices in batches:
                # Get batch data
                batch_states = []
                for idx in batch_indices:
                    batch_states.append(states[idx])
                
                # Prepare batch data
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Convert states to dict of batched tensors
                batched_states = {}
                for key in batch_states[0].keys():
                    batched_states[key] = torch.stack([torch.tensor(s[key], dtype=torch.float32) 
                                                    for s in batch_states]).to(self.device)
                
                # Forward pass
                policy_logits, values = self.model(batched_states)
                dist = Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratio and clipped loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO clip
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - CONFIG['ppo_clip'], 1.0 + CONFIG['ppo_clip']) * batch_advantages
                
                # Calculate losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Total loss
                loss = actor_loss + CONFIG['value_coef'] * critic_loss - CONFIG['entropy_coef'] * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear memory after update
        self.memory.clear()
    
    def train(self, num_episodes, checkpoint_dir):
        start_time = time.time()
        
        # Training loop
        episode_rewards = []
        best_reward = -float('inf')
        best_model_path = None
        
        for episode in range(1, num_episodes + 1):
            # Reset environment and get initial state
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Select action
                action, prob, value = self.select_action(state)
                
                # Execute action
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transition
                self.memory.store(state, action, prob, value, reward, done)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Learn when memory is full
                if len(self.memory.states) >= CONFIG['buffer_size']:
                    self.learn()
            
            # Always learn at the end of episode
            if len(self.memory.states) > 0:
                self.learn()
            
            # Track rewards
            episode_rewards.append(episode_reward)
            
            # Running average of rewards (last 100 episodes)
            if len(episode_rewards) > 100:
                avg_reward = sum(episode_rewards[-100:]) / 100
            else:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
            
            # Print progress
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Time: {elapsed_time:.2f}s")
            
            # Save checkpoint
            if episode % CONFIG['checkpoint_interval'] == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'episode': episode,
                    'reward': avg_reward
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'episode': episode,
                    'reward': best_reward
                }, best_model_path)
                print(f"New best model saved with reward {best_reward:.2f}!")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Best average reward: {best_reward:.2f}")
        
        return episode_rewards, best_model_path
    
    def test(self, model_path, num_episodes=10, render=True):
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Testing model: {model_path}")
        
        total_rewards = []
        success_count = 0
        
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                # Select best action (no exploration)
                with torch.no_grad():
                    states = {}
                    for key, value in state.items():
                        states[key] = torch.tensor([value], dtype=torch.float32).to(self.device)
                    
                    policy_logits, _ = self.model(states)
                    action = torch.argmax(policy_logits).item()
                
                # Execute action
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                steps += 1
                
                # Render if requested
                if render and episode <= 3:  # Only render first 3 episodes to save time
                    self.env.render()
            
            # Check if landing was successful
            ground_height = self.env.terrain[self.env.position]
            distance_to_ground = self.env.height - ground_height
            on_flat_area = any(start <= self.env.position < end for start, end, _ in self.env.flat_areas)
            
            success = distance_to_ground < 0.1 and on_flat_area
            if success:
                success_count += 1
            
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {steps}, Success = {success}")
            total_rewards.append(episode_reward)
        
        # Print summary
        avg_reward = sum(total_rewards) / num_episodes
        success_rate = success_count / num_episodes * 100
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Success rate: {success_rate:.2f}%")
        
        return avg_reward, success_rate

# Model Factory for easy model switching
class ModelFactory:
    @staticmethod
    def get_model(model_name, input_dims, action_dim):
        if model_name == "vision_transformer":
            return VisionTransformerModel(input_dims, action_dim)
        elif model_name == "resnet50":
            return ResNet50Model(input_dims, action_dim)
        elif model_name == "simple_cnn":
            return SimpleCNNModel(input_dims, action_dim)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

# Wrapper function to measure state dimensions
def get_input_dimensions(env):
    sample_state = env.reset()
    input_dims = {}
    
    for key, value in sample_state.items():
        if isinstance(value, np.ndarray):
            input_dims[key] = value.shape[0] if value.ndim == 1 else value.shape
    
    return input_dims

# Training function
def train_model(model_name, num_episodes=1000):
    print(f"Training model: {model_name}")
    
    # Create environment
    env = DroneLandingEnv()
    
    # Get input dimensions
    input_dims = get_input_dimensions(env)
    
    # Create model
    model = ModelFactory.get_model(model_name, input_dims, env.action_space.n)
    
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dirs(model_name)
    
    # Create agent
    agent = PPOAgent(model, env, device=CONFIG['device'])
    
    # Train
    rewards, best_model_path = agent.train(num_episodes, checkpoint_dir)
    
    # Test best model
    print("\nTesting best model:")
    avg_reward, success_rate = agent.test(best_model_path)
    
    return rewards, avg_reward, success_rate

# Model comparison function
def compare_models(model_names, num_episodes=1000):
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Training and testing {model_name}")
        print(f"{'='*50}")
        
        rewards, avg_reward, success_rate = train_model(model_name, num_episodes)
        
        results[model_name] = {
            'rewards': rewards,
            'avg_reward': avg_reward,
            'success_rate': success_rate
        }
    
    # Print comparison
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    print(f"{'Model':<20} | {'Average Reward':<20} | {'Success Rate':<20}")
    print("-" * 80)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} | {result['avg_reward']:>18.2f} | {result['success_rate']:>18.2f}%")
    
    return results

# Load and test a specific model
def load_and_test_model(model_name, model_path, num_episodes=10):
    # Create environment
    env = DroneLandingEnv()
    
    # Get input dimensions
    input_dims = get_input_dimensions(env)
    
    # Create model
    model = ModelFactory.get_model(model_name, input_dims, env.action_space.n)
    
    # Create agent
    agent = PPOAgent(model, env, device=CONFIG['device'])
    
    # Test model
    avg_reward, success_rate = agent.test(model_path, num_episodes, render=True)
    
    return avg_reward, success_rate

# Main function
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Drone Landing RL')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'compare'],
                        help='Mode: train, test, or compare')
    parser.add_argument('--model', type=str, default='vision_transformer',
                        choices=['vision_transformer', 'resnet50', 'simple_cnn'],
                        help='Model architecture to use')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training')
    parser.add_argument('--test_episodes', type=int, default=10,
                        help='Number of episodes for testing')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for testing')
    
    args = parser.parse_args()
    
    # Set device optimization for CPU
    if CONFIG['device'] == 'cpu':
        # Set number of threads for CPU optimization
        torch.set_num_threads(8)  # Adjust based on your CPU
    
    if args.mode == 'train':
        # Train a single model
        rewards, avg_reward, success_rate = train_model(args.model, args.episodes)
        print(f"Training completed. Final reward: {avg_reward:.2f}, Success rate: {success_rate:.2f}%")
    
    elif args.mode == 'test':
        # Test a pre-trained model
        if args.model_path is None:
            # If no specific path provided, use the best model from the model's checkpoint directory
            model_path = os.path.join(CONFIG['checkpoint_dir'], args.model, 'best_model.pt')
        else:
            model_path = args.model_path
        
        avg_reward, success_rate = load_and_test_model(args.model, model_path, args.test_episodes)
        print(f"Testing completed. Average reward: {avg_reward:.2f}, Success rate: {success_rate:.2f}%")
    
    elif args.mode == 'compare':
        # Compare different models
        models = ['vision_transformer', 'resnet50', 'simple_cnn']
        results = compare_models(models, args.episodes)
        
        # Visualize learning curves
        plt.figure(figsize=(12, 8))
        for model_name, result in results.items():
            # Plot smoothed rewards
            rewards = result['rewards']
            # Smooth the rewards for better visualization
            window_size = min(100, len(rewards) // 10)
            if window_size > 0:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label=model_name)
        
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.title('Learning Curves for Different Models')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curves.png')
        plt.show()

if __name__ == "__main__":
    main()