import gymnasium as gym
from Agent import Agent

def render_episode(agent, model_path, num_episodes):
    # Load the model
    agent.load_model(model_path)
    
    env = gym.make('CartPole-v1', render_mode='human')

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.reshape(1, -1)
        done = False
        total_reward = 0
        while not done:
            env.render()
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.reshape(1, -1)
            total_reward += reward
            state = next_state
            if truncated or terminated:
                done = True
        print(f"Episode {episode + 1} reward: {total_reward}")
    env.close()

# Initializing saved model
state_size = 4
action_size = 2
agent = Agent(
        state_size=state_size,
        action_size=action_size,
        n_hidden=24,
        memory_maxlen=5000,
        gamma=0.99,
        epsilon=0.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

# Load the model and render episodes
model_path = "RL/DQN/Models/FINAL_model.pt"
# model_path = "./cartpole_model/DQN_old.keras"
render_episode(agent, model_path, num_episodes=2)