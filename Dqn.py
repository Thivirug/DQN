import torch

class DQN(torch.nn.Module):
    """
        A DQN that takes "state_size" number of state information as input and maps them into state action values (Q) of size "action_size" 
    """
    def __init__(self, state_size: int, action_size: int, n_hidden: int):
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

    def forward(self, x):
        x = self.l1(x)
        x = self.activation1(x)
        x = self.l2(x)
        x = self.activation2(x)
        x = self.output(x)
        return x  