import torch

class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size, n_hidden):
        super().__init__()

        self.l1 = torch.nn.Linear(
            in_features = state_size,
            out_features = n_hidden
        )

        self.activation1 = torch.nn.ReLU()

        self.l2 = torch.nn.Linear(
            in_features = n_hidden,
            out_features = n_hidden
        )

        self.activation2 = torch.nn.ReLU()

        self.output = torch.nn.Linear(
            in_features = n_hidden,
            out_features = action_size
        )

        self.activation3 = torch.nn.Softmax()

    def forward(self, x):
        x = self.l1(x)
        x = self.activation1(x)
        x = self.l2(x)
        x = self.activation2(x)
        x = self.output(x)
        return self.activation3(x)