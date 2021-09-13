import torch
import torch.nn as nn


class MultiLayeredPerceptron(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[64], hidden_activation=nn.LeakyReLU(),
                 out_activation=nn.LeakyReLU()):
        super(MultiLayeredPerceptron, self).__init__()

        layers = []
        ins = [in_dim] + hidden_dims
        outs = hidden_dims + [out_dim]
        for i, o in zip(ins, outs):
            layers.append(nn.Linear(i, o))

        self.layers = nn.ModuleList(layers)

        activations = [hidden_activation] * len(hidden_dims)
        activations.append(out_activation)
        self.activations = nn.ModuleList(activations)

    def forward(self, x):
        for l, a in zip(self.layers, self.activations):
            x = l(x)
            x = a(x)
        return x


class MLP_sigmoid(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[64], hidden_activation=nn.LeakyReLU(),
                 out_activation=nn.Identity()):
        super(MLP_sigmoid, self).__init__()

        layers = []
        ins = [in_dim] + hidden_dims
        outs = hidden_dims + [out_dim]
        for i, o in zip(ins, outs):
            layers.append(nn.Linear(i, o))

        self.layers = nn.ModuleList(layers)

        activations = [hidden_activation] * len(hidden_dims)
        activations.append(out_activation)
        self.activations = nn.ModuleList(activations)

    def forward(self, x):
        for l, a in zip(self.layers, self.activations):
            x = l(x)
            x = a(x)

        return 1 / ((-x).exp() + 1)


if __name__ == '__main__':
    # MLP = MultiLayeredPerceptron(2, 10)
    tensor = torch.rand(5, 2)
