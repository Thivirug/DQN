# run the agent in the environment and train it
from Train import Trainer
from Agent import Agent
from Env import CartPole
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for plotting
import matplotlib.pyplot as plt
from Eval import render_episode

def plot_rewards(trainer_obj: Trainer):
    rewards_history = trainer_obj.total_rewards_history
    episodes = list(rewards_history.keys())
    rewards = list(rewards_history.values())

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, color='blue', marker='o', markersize=3)
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid()

    # save as image
    plt.savefig("RL/DQN/total_rewards_per_episode.png")
    plt.show()

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
        n_hidden=100,
        memory_maxlen=2000,
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
        n_episodes=2000,
        lr=0.001,
        optimizer="adam",
        target_update_freq=5,
        save_freq=100
    )

    # train the agent
    trainer.run_agent()

    # plot the rewards
    plot_rewards(trainer)

    # Load the model and render episodes
    model_path = "RL/DQN/Models/FINAL_model.pt"

    render_episode(agent, model_path, num_episodes=2)

