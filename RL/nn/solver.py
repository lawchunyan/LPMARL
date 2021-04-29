import torch
from optim.lp_solver import solve_LP
import pdb


class GraphMatchingSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = solve_LP(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: context for backpropagation
        :param grad_output: torch.Tensor of shape [batch_size, ?]
        :return:
        """
        pdb.set_trace()
        input, = ctx.saved_tensors
        grad_input_numpy = grad_output.detach().cpu().numpy()


if __name__ == '__main__':
    import torch.nn as nn
    from torch import autograd

    # debuging Solver
    rand_inp = torch.rand((16, 9))

    lin = nn.Linear(9, 1)
    Net = GraphMatchingSolver.apply

    out1 = lin(rand_inp)
    out1 = out1.reshape(4, 4)
    solution1 = Net(out1)

    print(autograd.grad(solution1.sum(), out1))

    # solution1.sum().backward()
