# run the agent in the environment and train it
from Train import Trainer
from Agent import Agent
import gymnasium as gym
import torch
from Env import CartPole

if __name__ == "__main__":
    # create the environment
    env = CartPole("CartPole-v1").env

    # get state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # create agent
    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        n_hidden=64,
        memory_maxlen=5000,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    # create trainer
    trainer = Trainer(
        agent=agent,
        env=env,
        batch_size=32,
        n_episodes=1000,
        lr=0.001,
        optimizer="adam",
        target_update_freq=10,
        save_freq=100
    )

    # train the agent
    trainer.run_agent()