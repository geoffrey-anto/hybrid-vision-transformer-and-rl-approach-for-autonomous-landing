import os
import time
import numpy as np
import torch
import airsim
import argparse
import logging
from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import logging
import random

# Add this near the top of the script, after the imports
def set_seeds(seed=100):
    """Set random seeds for reproducibility."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AirSimTest")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Test PPO Agent in AirSim")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="checkpoints/best_model.pth",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["vit", "resnet50", "cnn"], 
        default="vit",
        help="Type of feature extractor model to use"
    )
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=5,
        help="Number of test episodes to run"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=1000,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Save camera images during testing"
    )
    parser.add_argument(
        "--render_dir", 
        type=str, 
        default="renders",
        help="Directory to save rendered images"
    )
    return parser.parse_args()

def process_image(response):
    """Process AirSim image data into the correct format (3, 64, 64)."""
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    img_rgba = img1d.reshape(response.height, response.width, 3)
    img_rgb = img_rgba[:, :, :3]
    
    # Make sure the image is the correct size for the model (64x64)
    img_resized = np.array(Image.fromarray(img_rgb).resize((64, 64)))
    
    # Return as shape [3, 64, 64] for PyTorch (channels first)
    return np.transpose(img_resized, (2, 0, 1))

def process_lidar(lidar_data):
    """
    Process AirSim LiDAR data into a 16-element vector.
    Instead of using 3D coordinates, we'll take 16 distance readings.
    """
    # Extract point cloud from AirSim LiDAR data
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    
    # Calculate distances for each point (distance from sensor)
    if len(points) > 0:
        distances = np.sqrt(np.sum(points**2, axis=1))
        
        # If we have more than 16 points, sample 16 evenly around the drone
        if len(distances) >= 16:
            # Evenly sample 16 points
            indices = np.linspace(0, len(distances)-1, 16, dtype=int)
            lidar_readings = distances[indices]
        else:
            # Pad with max range if we have fewer than 16 points
            lidar_readings = np.ones(16, dtype=np.float32) * 100.0  # Default max range
            lidar_readings[:len(distances)] = distances
    else:
        # If no points, return max range for all directions
        lidar_readings = np.ones(16, dtype=np.float32) * 100.0
        
    return lidar_readings

def get_observation(client, vehicle_name="SimpleDrone"):
    """Get the current observation from AirSim."""
    try:
        # Get RGB camera image
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name)
        
        if not responses or len(responses) == 0:
            logger.error("Failed to get camera image")
            camera_data = np.zeros((3, 64, 64), dtype=np.float32)
        else:
            camera_data = process_image(responses[0])
        
        # Get LiDAR data
        lidar_data = client.getLidarData(lidar_name="Lidar", vehicle_name=vehicle_name)
        lidar_processed = process_lidar(lidar_data)
        
        # Verify shapes match what the model expects
        assert camera_data.shape == (3, 64, 64), f"Camera shape is {camera_data.shape}, expected (3, 64, 64)"
        assert lidar_processed.shape == (16,), f"LiDAR shape is {lidar_processed.shape}, expected (16,)"
        
        return {
            'camera': camera_data,
            'lidar': lidar_processed
        }
    except Exception as e:
        logger.error(f"Error getting observation: {e}")
        # Return empty observation as fallback
        return {
            'camera': np.zeros((3, 64, 64), dtype=np.float32),
            'lidar': np.ones(16, dtype=np.float32) * 100.0
        }

def take_action(client, action, vehicle_name="SimpleDrone"):
    """Execute the action in AirSim environment."""
    try:
        # Add random movement with 30% probability
        if np.random.random() < 0.3:
            # Randomly choose left or right
            random_action = np.random.choice([0, 1])  # 0 for left, 1 for right
            if random_action == 0:
                client.moveByVelocityAsync(-2.0, 0.0, 0.0, 1.0, vehicle_name=vehicle_name)
            else:
                client.moveByVelocityAsync(2.0, 0.0, 0.0, 1.0, vehicle_name=vehicle_name)
        else:
            # Execute the agent's chosen action
            if action == 0:  # Left
                client.moveByVelocityAsync(-1.0, 0.0, 0.0, 1.0, vehicle_name=vehicle_name)
            elif action == 1:  # Right
                client.moveByVelocityAsync(1.0, 0.0, 0.0, 1.0, vehicle_name=vehicle_name)
            elif action == 2:  # Down
                client.moveByVelocityAsync(0.0, 0.0, 3, 1.0, vehicle_name=vehicle_name)
        
        # Wait for the command to complete
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Error executing action: {e}")

def save_image(img_data, episode, step, render_dir):
    """Save a camera image during testing."""
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)
    
    # Convert from CHW to HWC and from numpy array to PIL Image
    img = np.transpose(img_data, (1, 2, 0))
    img = Image.fromarray(img.astype(np.uint8))
    
    # Save the image
    img.save(f"{render_dir}/episode_{episode}_step_{step}.png")

def test_agent(agent, args):
    """Test the agent in the AirSim environment."""
    # Connect to AirSim
    client = airsim.MultirotorClient()
    
    # Try to connect multiple times with delay
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            client.confirmConnection()
            logger.info("Connected to AirSim successfully")
            break
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.warning(f"Connection attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                logger.error(f"Failed to connect to AirSim after {max_attempts} attempts")
                return
    
    vehicle_name = "SimpleDrone"
    
    # Set up the render directory if needed
    if args.render:
        if not os.path.exists(args.render_dir):
            os.makedirs(args.render_dir)
    
    total_rewards = []
    
    for episode in range(args.num_episodes):
        logger.info(f"Starting episode {episode+1}/{args.num_episodes}")
        
        # Reset the environment
        try:
            client.reset()
            time.sleep(1)  # Give AirSim time to reset
            
            # Enable API control and arm
            client.enableApiControl(True, vehicle_name)
            client.armDisarm(True, vehicle_name)
            
            # Take off
            logger.info("Taking off...")
            client.takeoffAsync(vehicle_name=vehicle_name).join()
            
            # Move to initial position
            logger.info("Moving to initial position...")
            # Generate random x and y coordinates within ±50 meters
            random_x = np.random.uniform(-50, 50)
            random_y = np.random.uniform(-50, 50)
            client.moveToPositionAsync(random_x, random_y, -25, 5, vehicle_name=vehicle_name).join()

            
        except Exception as e:
            logger.error(f"Error in episode initialization: {e}")
            continue
        
        episode_reward = 0
        
        for step in range(args.max_steps):
            try:
                # Get the current observation
                observation = get_observation(client, vehicle_name)
                
                # Get action from agent
                action, _, _ = agent.get_action(observation, evaluate=True)
                
                # Execute the action
                take_action(client, action, vehicle_name)
                
                # Calculate reward (simplified for this example)
                reward = 1.0  # Simple reward for staying alive
                
                # Check collision
                collision_info = client.simGetCollisionInfo(vehicle_name)
                done = collision_info.has_collided
                
                if done:
                    reward = -100.0  # Penalty for collision
                    logger.info(f"Episode {episode+1} terminated due to collision at step {step}")
                    break
                
                episode_reward += reward
                
                # Save image if rendering is enabled
                if args.render and step % 10 == 0:  # Save every 10th frame to reduce disk usage
                    save_image(observation['camera'], episode+1, step, args.render_dir)
                
                # Print progress
                if step % 50 == 0:
                    logger.info(f"Episode {episode+1}, Step {step}, Current reward: {episode_reward}")
                    
            except Exception as e:
                logger.error(f"Error during step {step}: {e}")
                break
        
        total_rewards.append(episode_reward)
        logger.info(f"Episode {episode+1} finished with reward {episode_reward}")
        
        # Cleanup at the end of the episode
        try:
            client.armDisarm(False, vehicle_name)
            client.enableApiControl(False, vehicle_name)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # Calculate and print statistics
    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        std_reward = np.std(total_rewards)
        
        logger.info(f"Testing completed - Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"Rewards per episode: {total_rewards}")
    else:
        logger.warning("No episodes completed successfully")



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


def main():
    """Main function to run the test simulation."""
    args = parse_args()
    
    # Import your PPOAgent class
    try:
        agent = PPOAgent(model_type=args.model_type, device=device)
        logger.info(f"Successfully imported PPOAgent from current directory")
    except ImportError:
        # If that fails, try to import relative to script location
        try:
            import sys
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.append(script_dir)
            agent = PPOAgent(model_type=args.model_type, device=device)
            logger.info(f"Successfully imported PPOAgent from script directory")
        except ImportError:
            logger.error("Could not import PPOAgent class. Please make sure the file is in the correct location.")
            return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.warning(f"Model checkpoint not found at {args.model_path}. Using untrained agent.")
    else:
        # Load the model
        logger.info(f"Loading model from {args.model_path}")
        agent.load_checkpoint(args.model_path)
        
        # Log the model configuration
        logger.info(f"Model loaded successfully - Type: {agent.model_type}")
    
    # Test the agent
    test_agent(agent, args)
if __name__ == "__main__":
    main()