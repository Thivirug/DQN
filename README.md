# This project implements a Double DQN agent to solve Classic Control games in Gymnasium.

* The agent is trained using the Double DQN algorithm (***Coded from scratch***), which uses a neural network to approximate the Q-values for each action in a given state.
* The agent is trained on the Cartpole-v1 environment, which is a classic control problem where the goal is to balance a pole on a cart.
* The agent uses an epsilon-greedy policy for exploration, where it chooses a random action with probability epsilon and the action with the highest Q-value with probability 1-epsilon.
* The agent uses a replay buffer to store past experiences and sample them for training, which helps to break the correlation between consecutive experiences and stabilize training.
* The agent is trained for a specified number of episodes, and the total rewards for each episode are logged.
* The agent's performance is evaluated by rendering the environment and displaying the agent's actions in real-time.

# Usage
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```bash
   python Main.py
   ```
3. After training, run the evaluation script by copying the Agent object from `Main.py` and calling the `render_episode` function. Then,:
   ```bash
   python Eval.py
   ```
4. The trained models will be saved in the `Models` directory.
5. You can modify the model path in `Main.py` to load a different model for evaluation.
6. The training and evaluation results will be displayed in the console, and plots of the rewards will be generated using Matplotlib.
