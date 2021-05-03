import torch.nn as nn


class MultiLayeredPerceptron(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dims: list = [64, 64], hidden_activation=nn.LeakyReLU(),
                 out_actiation=nn.LeakyReLU()):
        super(MultiLayeredPerceptron, self).__init__()

        layers = []

        in_dims = [input_dim] + hidden_dims
        out_dims = hidden_dims + [out_dim]
        activations = [hidden_activation] * len(hidden_dims) + [out_actiation]

        for _in, _out, _act in zip(in_dims, out_dims, activations):
            layers.append(nn.Linear(_in, _out))
            layers.append(_act)

        self.layer = nn.Sequential(*layers)

    def forward(self, input):
        return self.layer(input)


if __name__ == '__main__':
    import torch
    MLP = MultiLayeredPerceptron(3, 4)
    _in = torch.rand((8, 3))
    out = MLP(_in)
