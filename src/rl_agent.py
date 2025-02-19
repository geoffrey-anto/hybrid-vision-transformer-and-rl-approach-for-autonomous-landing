import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from vision_transformer import VisionTransformer

class RLAgent(nn.Module):
    """
    A simple RL agent that uses a Vision Transformer backbone
    to extract features from images, then adds a small head
    to incorporate LiDAR stats and output actions.
    """
    def __init__(self, num_actions=4, lr=1e-4):
        super(RLAgent, self).__init__()
        # Vision transformer for image
        self.vit = VisionTransformer(num_classes=256)  # we use 256 as an intermediate embedding for RL

        # Additional head to incorporate LiDAR feature (1D) + 256 from ViT
        self.fc_fuse = nn.Sequential(
            nn.Linear(256 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, image, lidar_stat):
        """
        image: [batch_size, 3, 224, 224]
        lidar_stat: [batch_size, 1]
        """
        vit_out = self.vit(image)             # shape: [batch_size, 256]
        fused = torch.cat([vit_out, lidar_stat], dim=1)  # [batch_size, 257]
        action_logits = self.fc_fuse(fused)   # [batch_size, num_actions]
        return action_logits

    def get_action(self, image, lidar_stat):
        """
        Sample an action given the current state (stochastic policy).
        """
        with torch.no_grad():
            logits = self.forward(image, lidar_stat)
            # For a continuous action space, you might parameterize mean & std,
            # but here we'll treat the output as continuous direct values
            action = torch.tanh(logits)  # clamp actions between -1 and 1
        return action.cpu().numpy()

    def update_policy(self, states, actions, returns):
        """
        Policy gradient or other RL update. Simplified example below:
        states: list of (image, lidar), each image shape => [3, 224, 224]
        actions: array of shape [batch_size, num_actions]
        returns: array of shape [batch_size]
        """
        # This is a rudimentary policy gradient approach
        self.optimizer.zero_grad()

        # Convert to torch
        images_t = torch.stack([torch.tensor(s[0], dtype=torch.float32) for s in states])
        lidar_t = torch.stack([torch.tensor(s[1], dtype=torch.float32) for s in states])
        # Add batch dimension
        images_t = images_t.to(next(self.parameters()).device)
        lidar_t = lidar_t.to(next(self.parameters()).device)
        # actions, returns
        actions_t = torch.tensor(actions, dtype=torch.float32, device=images_t.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=images_t.device)

        # forward pass
        logits = self.forward(images_t, lidar_t)
        # For simplicity, we treat each dimension as an independent distribution center
        # We'll compute MSE loss to the actions weighted by returns (very non-standard).
        # A real approach would parameterize a distribution with log probs or so.
        loss = ((logits - actions_t) ** 2 * returns_t.unsqueeze(1)).mean()

        loss.backward()
        self.optimizer.step()
        return loss.item()
