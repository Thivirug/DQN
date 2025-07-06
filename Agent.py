from Dqn import DQN
import numpy as np
from collections import deque

class Agent:
    def __init__(self, state_size: int, action_size: int, n_hidden: int, memory_maxlen: int):
        # model args
        self.state_size = state_size,
        self.action_size = action_size,
        self.n_hidden = n_hidden

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
    
    def store(self, curr_state, action, reward, nxt_state, done):
        """
            Update experience replay.
        """
        self.exp_replay.append(curr_state, action, reward, nxt_state, done)


    

    

    
