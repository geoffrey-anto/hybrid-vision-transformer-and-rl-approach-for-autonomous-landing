import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
from gym import spaces
import airsim
import cv2
from PIL import Image
from transformers import ViTModel, ViTConfig
import random
from collections import deque

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
    "checkpoint_interval": 10000,  # Save model every n steps
    "eval_interval": 5000,  # Evaluate model every n steps
    "checkpoint_dir": "./checkpoints",
    "model_name": "drone_landing_model",
    "lidar_sensor_name": "LidarSensor1"  # Name of the LiDAR sensor in AirSim
}

# Create checkpoint directory
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

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

class AirSimDroneLandingEnv:
    """Custom environment for drone landing in AirSim"""
    
    def __init__(self):
        # Connect to AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Set up the LiDAR sensor
        self.setup_lidar_sensor()
        
        # Action and observation spaces
        # Actions: Move Forward, Backward, Left, Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: Camera image and LiDAR points
        self.observation_space = spaces.Dict({
            'camera': spaces.Box(low=0, high=255, shape=(3, CONFIG["image_size"], CONFIG["image_size"]), dtype=np.uint8),
            'lidar': spaces.Box(low=-100, high=100, shape=(CONFIG["lidar_points"], 3), dtype=np.float32)
        })
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 500
        
    def setup_lidar_sensor(self):
        """Set up the LiDAR sensor in AirSim"""
        try:
            # Check if the LiDAR sensor is already set up
            self.client.getLidarData(lidar_name="LidarSensor2", vehicle_name="")
            print(f"LiDAR sensor '{CONFIG['lidar_sensor_name']}' already exists.")
        except:
            print(f"Setting up LiDAR sensor '{CONFIG['lidar_sensor_name']}'...")
            # If using settings.json approach, just print instructions
            print("Please ensure your AirSim settings.json includes a LiDAR sensor configuration:")
            print("""
            "Sensors": {
                "LidarSensor1": {
                    "SensorType": 6,
                    "Enabled": true,
                    "NumberOfChannels": 16,
                    "RotationsPerSecond": 10,
                    "PointsPerSecond": 10000,
                    "X": 0, "Y": 0, "Z": -1,
                    "Roll": 0, "Pitch": 0, "Yaw": 0,
                    "VerticalFOVUpper": 10,
                    "VerticalFOVLower": -10,
                    "HorizontalFOVStart": -45,
                    "HorizontalFOVEnd": 45,
                    "DrawDebugPoints": true,
                    "DataFrame": "SensorLocalFrame"
                }
            }
            """)
    
    def reset(self):
        """Reset the environment at the beginning of an episode"""
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Take off and hover at starting position (about 20m above ground)
        self.client.takeoffAsync().join()
        
        # Random starting position within a reasonable area
        x = np.random.uniform(-40, 40)
        y = np.random.uniform(-40, 40)
        print(x, y)
        z = -30  # 20m above ground (negative z is up in AirSim)
        
        # Move to starting position
        self.client.moveToPositionAsync(x, y, z, 5).join()
        self.client.hoverAsync().join()
        
        # Wait a moment for sensors to stabilize
        time.sleep(1)
        
        # Reset episode steps
        self.episode_steps = 0
        
        # Get initial observation
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        self.episode_steps += 1
        
        # Convert action to drone movement
        if action == 0:  # Left
            self.client.moveByVelocityAsync(0, -1, 0, 0.5).join()
        elif action == 1:  # Right
            self.client.moveByVelocityAsync(0, 1, 0, 0.5).join()
        elif action == 2:  # Down
            self.client.moveByVelocityAsync(0, 0, 1, 0.5).join()
        elif action == 3:  # Forward
            self.client.moveByVelocityAsync(1, 0, 0, 0.5).join()
        
        # Wait a bit for the action to take effect
        time.sleep(0.1)
        
        # Get drone state
        drone_state = self.client.getMultirotorState()
        position = drone_state.kinematics_estimated.position

        if self.episode_steps % 10 == 0:
            print(f"Step {self.episode_steps}: Position: ({position.x_val}, {position.y_val}, {position.z_val})")
        
        # Calculate distance to ground
        ground_z = 0  # Assuming ground is at z=0
        height_above_ground = abs(position.z_val)
        
        # Check for landing or collision
        collision_info = self.client.simGetCollisionInfo()
        landed = height_above_ground < 0.5 and abs(drone_state.kinematics_estimated.linear_velocity.z_val) < 0.1
        crashed = collision_info.has_collided and not landed
        
        # Determine if episode is done
        done = landed or crashed or self.episode_steps >= self.max_episode_steps
        
        # Calculate reward
        reward = self._compute_reward(height_above_ground, landed, crashed)
        
        # Get observation
        obs = self._get_observation()
        
        # Info dictionary
        info = {
            'height': height_above_ground,
            'landed': landed,
            'crashed': crashed,
            'position': (position.x_val, position.y_val, position.z_val)
        }
        
        return obs, reward, done, info
    
    def _compute_reward(self, height, landed, crashed):
        """Compute reward based on drone state"""
        if crashed:
            return -100  # Large penalty for crashing
        
        if landed:
            return 100  # Large reward for successful landing
        
        # Encourage descending, but not too fast
        drone_state = self.client.getMultirotorState()
        velocity_z = drone_state.kinematics_estimated.linear_velocity.z_val
        
        # Small reward for getting closer to the ground
        height_reward = -0.2 * height
        
        # Penalty for high velocity when close to ground
        velocity_penalty = 0
        if height < 5 and velocity_z > 1:
            velocity_penalty = -0.5 * (velocity_z - 1)**2
        
        # Encourage the drone to stay level
        orientation = drone_state.kinematics_estimated.orientation
        pitch, roll = airsim.to_eularian_angles(orientation)[0:2]
        orientation_penalty = -0.1 * (abs(pitch) + abs(roll))
        
        return height_reward + velocity_penalty + orientation_penalty
    
    def _get_observation(self):
        """Get observation from AirSim"""
        # Get camera image
        responses = self.client.simGetImages([
            airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)
        ])
        
        # Process image
        img_rgba = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgba = img_rgba.reshape(responses[0].height, responses[0].width, 3)
        img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2RGB)
        
        # Resize image to match ViT input size
        img = cv2.resize(img_rgb, (CONFIG["image_size"], CONFIG["image_size"]))
        img = np.transpose(img, (2, 0, 1))  # Convert to channel-first format
        
        # Get LiDAR data with error handling
        try:
            lidar_data = self.client.getLidarData(lidar_name="LidarSensor2")
            
            if lidar_data and len(lidar_data.point_cloud) >= 3:
                # Process valid LiDAR data
                points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            else:
                # Handle empty LiDAR data
                # print("Warning: Empty LiDAR data received, using simulated data")
                # Generate fake points around the drone as a fallback
                drone_pos = self.client.getMultirotorState().kinematics_estimated.position
                points = self._generate_simulated_lidar_points(drone_pos)
        except Exception as e:
            print(f"Error getting LiDAR data: {e}")
            print("Using simulated LiDAR data instead")
            # Generate synthetic LiDAR points as fallback
            drone_pos = self.client.getMultirotorState().kinematics_estimated.position
            points = self._generate_simulated_lidar_points(drone_pos)
        
        # If we have too few points, pad with zeros
        if len(points) < CONFIG["lidar_points"]:
            padding = np.zeros((CONFIG["lidar_points"] - len(points), 3), dtype=np.float32)
            points = np.vstack([points, padding])
        # If we have too many points, sample randomly
        elif len(points) > CONFIG["lidar_points"]:
            indices = np.random.choice(len(points), CONFIG["lidar_points"], replace=False)
            points = points[indices]
            
        return {
            'camera': img.astype(np.float32) / 255.0,  # Normalize to [0,1]
            'lidar': points
        }
    
    def _generate_simulated_lidar_points(self, drone_pos, num_points=100):
        """Generate synthetic LiDAR points for testing when real data isn't available"""
        # Get drone position
        x, y, z = drone_pos.x_val, drone_pos.y_val, drone_pos.z_val
        
        # Generate points in a cone shape pointing downward
        points = []
        for _ in range(num_points):
            # Random angle in 360 degrees
            angle = np.random.uniform(0, 2 * np.pi)
            # Random distance from center, increasing with radius (cone shape)
            dist_factor = np.random.uniform(0, 1) ** 0.5  # Square root for more even distribution
            dist_h = dist_factor * abs(z) * 0.5  # Horizontal distance proportional to height
            
            # Calculate point coordinates (cone pointing down from drone)
            px = x + dist_h * np.cos(angle)
            py = y + dist_h * np.sin(angle)
            pz = np.random.uniform(z, 0)  # Between drone height and ground
            
            points.append([px, py, pz])
        
        return np.array(points, dtype=np.float32)

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
            torch.BoolTensor(self.dones),
            indices
        )
        
    def clear(self):
        self.size = 0

def compute_gae(rewards, values, dones, gamma, lam):
    """Compute Generalized Advantage Estimation"""
    gae = 0
    returns = np.zeros_like(values)
    
    for t in reversed(range(len(rewards))):
        if t < len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = 0
            
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        returns[t] = gae + values[t]
        
    return returns

def train():
    """Main training function"""
    # Create environment and policy
    env = AirSimDroneLandingEnv()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy = ViTDronePolicy(
        image_size=CONFIG["image_size"],
        patch_size=CONFIG["patch_size"],
        lidar_points=CONFIG["lidar_points"],
        hidden_dim=CONFIG["hidden_dim"],
        action_dim=env.action_space.n
    ).to(device)
    
    optimizer = optim.Adam(policy.parameters(), lr=CONFIG["learning_rate"])
    
    # Experience buffer
    buffer = Buffer(
        CONFIG["buffer_size"],
        CONFIG["image_size"],
        CONFIG["lidar_points"]
    )
    
    # Tracking variables
    global_step = 0
    episode_rewards = []
    
    print("Starting training...")
    
    try:
        while global_step < CONFIG["total_timesteps"]:
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done and global_step < CONFIG["total_timesteps"]:
                # Convert observations to tensors
                camera_img = torch.FloatTensor(np.expand_dims(obs['camera'], 0)).to(device)
                lidar_data = torch.FloatTensor(np.expand_dims(obs['lidar'], 0)).to(device)
                
                # Get action from policy
                action, log_prob, value = policy.act(camera_img, lidar_data)
                
                # Step environment
                next_obs, reward, done, info = env.step(action)
                
                # Store transition in buffer
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
                episode_reward += reward
                global_step += 1
                
                # Check if buffer is full and update policy
                if buffer.size == CONFIG["buffer_size"]:
                    # Get data from buffer
                    camera_imgs, lidar_data, actions, rewards, values, old_log_probs, dones, indices = buffer.get()
                    
                    # Compute returns using GAE
                    returns = compute_gae(
                        rewards.numpy(), 
                        values.numpy(), 
                        dones.numpy(), 
                        CONFIG["gamma"], 
                        CONFIG["gae_lambda"]
                    )
                    returns = torch.FloatTensor(returns)
                    
                    # PPO update
                    for _ in range(CONFIG["ppo_epochs"]):
                        # Generate random mini-batches
                        batch_size = CONFIG["batch_size"]
                        batch_indices = np.random.choice(
                            CONFIG["buffer_size"], 
                            batch_size, 
                            replace=False
                        )
                        
                        # Get batch data
                        batch_camera_imgs = camera_imgs[batch_indices].to(device)
                        batch_lidar_data = lidar_data[batch_indices].to(device)
                        batch_actions = actions[batch_indices].to(device)
                        batch_returns = returns[batch_indices].to(device)
                        batch_old_log_probs = old_log_probs[batch_indices].to(device)
                        
                        # Forward pass
                        new_log_probs, entropy, new_values = policy.evaluate(
                            batch_camera_imgs,
                            batch_lidar_data,
                            batch_actions
                        )
                        
                        # Calculate ratio and surrogate loss
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        advantages = batch_returns - new_values.squeeze(-1)
                        
                        # Normalize advantages
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        
                        # PPO policy loss
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1.0 - CONFIG["clip_param"], 1.0 + CONFIG["clip_param"]) * advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        value_loss = CONFIG["value_loss_coef"] * torch.nn.functional.mse_loss(new_values.squeeze(-1), batch_returns)
                        
                        # Entropy bonus
                        entropy_loss = -CONFIG["entropy_coef"] * entropy.mean()
                        
                        # Total loss
                        loss = policy_loss + value_loss + entropy_loss
                        
                        # Update policy
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy.parameters(), CONFIG["max_grad_norm"])
                        optimizer.step()
                    
                    # Clear buffer
                    buffer.clear()
                
                # Save checkpoint periodically
                if global_step % CONFIG["checkpoint_interval"] == 0:
                    checkpoint_path = f"{CONFIG['checkpoint_dir']}/{CONFIG['model_name']}_step_{global_step}.pt"
                    torch.save({
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step
                    }, checkpoint_path)
                    print(f"Saved checkpoint at step {global_step} to {checkpoint_path}")
                
                # Evaluate model periodically
                if global_step % CONFIG["eval_interval"] == 0:
                    eval_reward = evaluate(policy, env, device, episodes=5)
                    print(f"Step {global_step}: Evaluation reward: {eval_reward:.2f}")
            
            # Track episode rewards
            episode_rewards.append(episode_reward)
            print(f"Episode finished. Reward: {episode_reward:.2f}, Total steps: {global_step}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
    finally:
        # Save final model
        final_checkpoint_path = f"{CONFIG['checkpoint_dir']}/{CONFIG['model_name']}_final.pt"
        torch.save({
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, final_checkpoint_path)
        print(f"Training complete. Final model saved to {final_checkpoint_path}")

def evaluate(policy, env, device, episodes=10):
    """Evaluate the policy without exploration"""
    policy.eval()
    total_rewards = []
    
    for _ in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Convert observations to tensors
            camera_img = torch.FloatTensor(np.expand_dims(obs['camera'], 0)).to(device)
            lidar_data = torch.FloatTensor(np.expand_dims(obs['lidar'], 0)).to(device)
            
            # Get best action (no exploration)
            with torch.no_grad():
                action_logits, _ = policy(camera_img, lidar_data)
                action = torch.argmax(action_logits, dim=1).item()
            
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    policy.train()
    return np.mean(total_rewards)

def test(checkpoint_path):
    """Test the trained model"""
    # Create environment and policy
    env = AirSimDroneLandingEnv()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy = ViTDronePolicy(
        image_size=CONFIG["image_size"],
        patch_size=CONFIG["patch_size"],
        lidar_points=CONFIG["lidar_points"],
        hidden_dim=CONFIG["hidden_dim"],
        action_dim=env.action_space.n
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Testing for 10 episodes...")
    
    # Test for multiple episodes
    successful_landings = 0
    total_episodes = 10
    
    for episode in range(total_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        print(f"Episode {episode+1}/{total_episodes}")
        
        while not done:
            # Convert observations to tensors
            camera_img = torch.FloatTensor(np.expand_dims(obs['camera'], 0)).to(device)
            lidar_data = torch.FloatTensor(np.expand_dims(obs['lidar'], 0)).to(device)
            
            # Get best action (no exploration)
            with torch.no_grad():
                action_logits, _ = policy(camera_img, lidar_data)
                action = torch.argmax(action_logits, dim=1).item()
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Print status every 10 steps
            if step % 10 == 0:
                print(f"  Step {step}: Height: {info['height']:.2f}m, Reward: {reward:.2f}")
        
        # Check if landing was successful
        if info.get('landed', False):
            print(f"Episode {episode+1} - SUCCESS: Landed safely! Total reward: {episode_reward:.2f}")
            successful_landings += 1
        else:
            print(f"Episode {episode+1} - FAILURE: Mission failed. Total reward: {episode_reward:.2f}")
    
    success_rate = successful_landings / total_episodes * 100
    print(f"\nTest completed. Success rate: {success_rate:.2f}% ({successful_landings}/{total_episodes})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Drone Landing with RL")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                       help="Whether to train or test the model")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint path for testing")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "test":
        if args.checkpoint is None:
            print("Please provide a checkpoint path for testing with --checkpoint")
        else:
            test(args.checkpoint)