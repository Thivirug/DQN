from Agent import Agent
import gymnasium
import torch
import os
import numpy as np

class Trainer:
    def __init__(self, agent: Agent, env: gymnasium.Env, batch_size: int, n_episodes: int, lr: float, optimizer: torch.optim.Optimizer, target_update_freq: int, save_freq: int, model_save_path: str = "RL/DQN/Models"):
        self.agent = agent
        self.env = env
        self.bs = batch_size
        self.n_eps = n_episodes
        self.freq = target_update_freq
        self.model_file = model_save_path
        self.save_freq = save_freq
        self.lr = lr
        self.optimizer = optimizer
        self.total_rewards_history = {}
        
        # Initialize optimizer
        if self.optimizer == "adam":
            self.opt = torch.optim.Adam(self.agent.main_model.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            self.opt = torch.optim.SGD(self.agent.main_model.parameters(), lr=self.lr)
            
    def train(self):
        """
            Train the agent using experience replay.
        """
        # get a batch of exp replays
        exp_batch = self.agent.replay(self.bs)

        # print(f"exp_batch type: {type(exp_batch)}")
        # print(f"exp_batch length: {len(exp_batch)}")
        # print(f"exp_batch[0] type: {type(exp_batch[0])}")
        # print(f"exp_batch[0] shape/length: {exp_batch[0].shape if hasattr(exp_batch[0], 'shape') else len(exp_batch[0])}")
    

        for curr_state, action, reward, nxt_state, done in exp_batch:
            # zero grad
            self.opt.zero_grad()

            # get Q value using main network 
            Q_v_main = self.agent.main_model(curr_state)[0][action]

            # calculate true Q value using target network (y_true)
            if done: 
                target = reward

            else:
                T = torch.max(self.agent.target_model(nxt_state))
                target = reward + self.agent.gamma * T

            # make target tensor
            target_tensor = torch.tensor(target, dtype=torch.float32)
            # calc loss
            loss = torch.nn.functional.mse_loss(Q_v_main, target_tensor)
            # backprop
            loss.backward()

            self.opt.step()

    def run_agent(self):
        """
            Run the agent in env.
        """

        for ep in range(1,self.n_eps):
            # reset env
            state = self.env.reset()[0]
            
            # convert state to np array
            state = np.reshape(state, (1, -1))  # Ensure state is 2D for DQN input
            
            total_reward = 0

            # --- agent's journey starts ---
            n = 500
            for t in range(n):
                # take an action
                a = self.agent.act(state)
                # update agent's env perception
                obs, reward, terminated, truncated, _ = self.env.step(a)

                done = terminated or truncated

                # update agent's memory (exp replay)
                next_state = np.reshape(obs, (1, -1))
                self.agent.remember(state, a, reward, next_state, done)

                # update state
                state = next_state

                total_reward += reward

                # if terminated or truncated - print episode info and skip to next episode
                if done:
                    print(f"------ Episode terminated? {terminated}, truncated? {truncated} -------")
                    print(f"Episode: {ep} / {self.n_eps}, Score: {t} / {n}, Epsilon: {self.agent.epsilon:.4f}")
                    break

            # append total reward to dict 
            self.total_rewards_history[ep] = total_reward

            if ep % 50 == 0:
                print(f"\n\tEpisode: {ep}, Total Reward: {total_reward}\n")

            # train main agent DQN
            if len(self.agent.exp_replay) > self.bs:   # training only when there's exp more than batch size
                self.train()

            # decay epsilon per episode
            self.agent.eps_decay()

            # update target network every specified times
            if ep % self.freq == 0:
                self.agent._update_target_model()

            # save model every specified num of episodes
            if ep % self.save_freq == 0:
                path = os.path.join(self.model_file, f"model_EP_{ep}.pt")
                self.agent.save_model(path)

        # final model save
        final_path = os.path.join(self.model_file, "FINAL_model.pt")
        self.agent.save_model(final_path)
        


