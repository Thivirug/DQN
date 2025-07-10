import gymnasium as gym
from Agent import Agent

def render_episode(agent, model_path, num_episodes, env_name):
    # Load the model
    agent.load_model(model_path)
    
    # render env
    env = gym.make(env_name, render_mode='human')

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

# ! ------------------------------------------ CARTPOLE ----------------------------------------------

# if __name__ == "__main__":
#     # get state and action sizes # ! Change hard coded values if a new environment is used
#     state_size = 4
#     action_size = 2

#     # create agent # ! Make sure to copy the exact agent instance used for training. epsilon value can be changed.
#     agent = Agent(
#         state_size=state_size,
#         action_size=action_size,
#         n_hidden=100,
#         memory_maxlen=2000,
#         gamma=0.99,
#         epsilon=0, # exploitation only during evaluation
#         epsilon_min=0.01,
#         epsilon_decay=0.995
#     )

#     # Load the model and render episodes
#     ENV_NAME = 'CartPole-v1'
#     model_path = f"RL/DQN/Models/{ENV_NAME}/FINAL_model.pt"

#     render_episode(agent, model_path, num_episodes=2, env_name=ENV_NAME)

# ! ------------------------------------------ Mountain CAR ----------------------------------------------

if __name__ == "__main__":
    # get state and action sizes # ! Change hard coded values if a new environment is used
    state_size = 2
    action_size = 3

    # create agent # ! Make sure to copy the exact agent instance used for training. epsilon value can be changed.
    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        n_hidden=200,
        memory_maxlen=2000,
        gamma=0.99,
        epsilon=0,
        epsilon_min=0.001,
        epsilon_decay=0.995
    )

    # Load the model and render episodes
    ENV_NAME = 'MountainCar-v0'
    model_path = f"RL/DQN/Models/{ENV_NAME}/FINAL_model.pt"

    render_episode(agent, model_path, num_episodes=2, env_name=ENV_NAME)
