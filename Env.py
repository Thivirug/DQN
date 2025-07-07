import gymnasium as gym

class CartPole():
    def __init__(self, name):
        self.name = name
        self.env = self.make_env()

    def make_env(self):
        return gym.make(self.name)

    def info(self):
        print("States: ", self.env.observation_space)
        print("State size: ", self.env.observation_space.shape[0])
        print("Actions: ", self.env.action_space)
        print("Action size: ", self.env.action_space.n)

if __name__ == "__main__":
    env = CartPole("CartPole-v1")
    env.info()
