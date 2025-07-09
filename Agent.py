from Dqn import DQN
import numpy as np
from collections import deque
import random
import torch
import os

class Agent:
    def __init__(self, state_size: int, action_size: int, n_hidden: int, memory_maxlen: int, gamma: float, epsilon: float, epsilon_min: float, epsilon_decay: float):
        # model args
        self.state_size = state_size  
        self.action_size = action_size  
        self.n_hidden = n_hidden

        # algorithm args
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # experience replay 
        self.exp_replay = deque(maxlen = memory_maxlen)

        # main DQN network 
        self.main_model = self.build_model()
        # target DQN network
        self.target_model = self.build_model()

    def build_model(self):
        """
            Create the deep Q network.
        """
        return DQN(self.state_size, self.action_size, self.n_hidden)
    
    def _update_target_model(self):
        """
            Copies weights of main network to target network.
        """
        self.target_model.load_state_dict(self.main_model.state_dict())
    
    def remember(self, curr_state, action, reward, nxt_state, done: bool):
        """
            Update experience replay.
        """
        # convert to tensor
        curr_state = torch.tensor(curr_state, dtype=torch.float32)
        nxt_state = torch.tensor(nxt_state, dtype=torch.float32)

        # append to exp replay
        self.exp_replay.append((curr_state, action, reward, nxt_state, done))

    def save_model(self, filepath: os.PathLike):
        """
            Save DQN.
        """
        torch.save(self.main_model.state_dict(), filepath)


    def load_model(self, filepath: os.PathLike):
        """
            Load DQN.
        """
        return self.main_model.load_state_dict(torch.load(filepath, weights_only=True))

    def act(self, state: np.ndarray):
        """
            Choose an action based on epsilon greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            # explore
            return random.randrange(self.action_size)
        
        # exploit
        state_tensor = torch.Tensor(state)
        # print(f"State tensor: {state_tensor}")  # Debugging line to check state tensor
        q_values = self.main_model(state_tensor)
        # print(f"Q-values: {q_values}")  # Debugging line to check Q-values
        return torch.argmax(q_values[0]).item()  # !
    

    def replay(self, batch_size: int):
        """
            Return a batch of sample exp tuples from experience replay.
        """
        # sample exp tuples batch from memory
        batch = random.sample(self.exp_replay, batch_size)

        return batch
        
    def eps_decay(self):
        """
            Decay epsilon.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


            










