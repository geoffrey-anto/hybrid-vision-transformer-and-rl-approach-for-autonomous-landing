import torch
import numpy as np
import random
from environment import AirSimLandingEnv
from rl_agent import RLAgent

def run_episode(env, agent, max_steps=200, train=False, gamma=0.99):
    """
    Runs one episode in the environment.
    If train=True, we store transitions for policy updates.
    """
    states = []
    actions = []
    rewards = []

    # Reset environment
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < max_steps:
        step_count += 1
        image_np, lidar_np = obs
        # Add batch dimension
        image_t = torch.from_numpy(image_np).unsqueeze(0).float().cuda()
        lidar_t = torch.from_numpy(lidar_np).unsqueeze(0).float().cuda()

        action_t = agent.get_action(image_t, lidar_t)[0]  # [4] in numpy
        next_obs, reward, done, info = env.step(action_t)

        if train:
            # store transition
            states.append(obs)
            actions.append(action_t)
            rewards.append(reward)

        obs = next_obs
        total_reward += reward

    if train:
        # compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # policy update
        loss = agent.update_policy(states, actions, returns)
        return total_reward, loss
    else:
        return total_reward, None

def main():
    # Create environment
    env = AirSimLandingEnv()
    # Put agent on GPU if available
    agent = RLAgent(num_actions=4).cuda()

    num_episodes = 10  # Increase for real training
    for ep in range(num_episodes):
        total_reward, loss = run_episode(env, agent, max_steps=200, train=True)
        print(f"Episode {ep+1}/{num_episodes}, Total Reward: {total_reward}, Loss: {loss}")

    # Test run
    test_reward, _ = run_episode(env, agent, max_steps=200, train=False)
    print(f"Test Episode Reward: {test_reward}")

    env.close()

if __name__ == "__main__":
    main()
