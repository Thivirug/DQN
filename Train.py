from Agent import Agent
import gymnasium
import torch
import os

class Trainer:
    def __init__(self, agent: Agent, env: gymnasium.Env, batch_size: int, n_episodes: int, lr: float, optimizer: torch.optim.Optimizer, target_update_freq: int, save_freq: int, model_save_path: str = "Models"):
        self.agent = agent
        self.env = env
        self.bs = batch_size
        self.n_eps = n_episodes
        self.freq = target_update_freq
        self.model_file = model_save_path
        self.save_freq = save_freq
        self.lr = lr
        self.optimizer = optimizer
        
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

        for curr_state, action, reward, nxt_state, done in exp_batch:
            # zero grad
            self.opt.zero_grad()

            # get Q value using main network (y_pred)
            Q_v_pred = self.agent.main_model(curr_state.unsqueeze(0))[0][action]

            # calculate true Q value using target network (y_true)
            if done: 
                Q_v_true = reward

            else:
                T = torch.max(self.agent.target_model(nxt_state.unsqueeze(0)))[0].item()
                Q_v_true = reward + self.agent.gamma * T

            # calc loss
            loss = torch.nn.functional.mse_loss(Q_v_pred, Q_v_true)
            # backprop
            loss.backward()

            self.opt.step()


    def run_agent(self):
        """
            Run the agent in env.
        """

        for ep in self.n_eps:
            # reset env
            state = self.env.reset()[0]
            total_reward = 0

            # --- agent's journey starts ---
            for t in range(500):
                # take an action
                a = self.agent.act(state)
                # update agent's env perception
                obs, reward, terminated, truncated, _ = self.env.step(a)

                done = terminated or truncated

                # update agent's memory (exp replay)
                next_state = obs[0]
                self.agent.remember(state, a, reward, next_state, done)

                # update state
                state = next_state

                total_reward += reward

                # if terminated or truncated - print episode info and skip to next episode
                if done:
                    print(f"Episode: {ep / self.n_eps}, Score: {t}, Epsilon: {self.agent.epsilon:.4f}")
                    break

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
        


