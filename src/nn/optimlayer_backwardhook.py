import cvxpy as cp
import torch
import numpy as np
import pdb
import torch.nn as nn
import torch.autograd as autograd
from itertools import accumulate

default_solving_dict = {
    'max_iters': 100,
    'abstol': 1e-8,
    'reltol': 1e-8,
    'feastol': 1e-8, }

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def objective(x, coeff, A, b, C, d):
    return - coeff @ x + 1e-5 * sum(x * x)  # ok


def equality(x, coeff, A, b, C, d):
    return A @ x - b


def inequality(x, coeff, A, b, C, d):
    return C @ x - d


def ineq_lb(x, coeff, A, b, C, d):
    return - x


def ineq_up(x, coeff, A, b, C, d):
    return x - 1


class MatchginSolver():
    def __init__(self, n, m, c, device='cpu'):
        x = cp.Variable(n * m)
        coeff = cp.Parameter(n * m)
        A = cp.Parameter((n, n * m))
        b = cp.Parameter(n)
        C = cp.Parameter((m, n * m))
        d = cp.Parameter(m)

        self.variables = [x]
        self.problem_parameters = [coeff, A, b, C, d]
        inequalities = [inequality, ineq_lb, ineq_up]
        equalities = [equality]

        cp_equalities = [eq(*self.variables, *self.problem_parameters) == 0 for eq in equalities]
        cp_inequalities = [ineq(*self.variables, *self.problem_parameters) <= 0 for ineq in inequalities]
        self.problem = cp.Problem(cp.Minimize(objective(*self.variables, *self.problem_parameters)),
                                  cp_inequalities + cp_equalities)
        self.initialize_problem(n, m, c, device)

    def initialize_problem(self, n, m, coeff, device='cpu'):
        A_val = np.zeros((n, n * m))
        for i in range(n):
            A_val[i, i * m:(i + 1) * m] = 1
        A_val = A_val
        b_val = np.ones(n)

        # set C value
        temp = np.eye(m)
        C_value = np.concatenate([temp for _ in range(n)], axis=-1)
        C_val = C_value
        d_val = np.ones(m) * coeff
        default_parameters = [A_val, b_val, C_val, d_val]
        self.default_parameters = default_parameters

    def solve(self, coeff):
        params = [coeff] + self.default_parameters
        # with torch.no_grad():
        for i, p in enumerate(self.problem_parameters):
            # p.value = params[i].cpu().double().numpy()
            p.value = params[i]
        self.problem.solve()
        out = self.variables[0].value
        return out

solver = MatchginSolver(2, 2, 1.1, device)

# class EdgeMatching(nn.Module):
#     def __init__(self):
#         super(EdgeMatching, self).__init__()
#         self.matchinglayer = EdgeMatching_autograd()
#
#     def forward(self, coeff):
#         return self.matchinglayer.apply(coeff)

lambda_val = 50


class EdgeMatching_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_cost):
        ctx.edge_cost = edge_cost.detach().cpu().numpy()
        # ctx.solver = solver
        ctx.sol = solver.solve(ctx.edge_cost)
        return torch.from_numpy(ctx.sol).float().to(edge_cost.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.sol.shape
        device = grad_output.device
        grad_output = grad_output.cpu().numpy()

        edge_coest_prime = ctx.edge_cost + lambda_val * grad_output
        better_sol = solver.solve(coeff=edge_coest_prime)

        gradient = -(ctx.sol - better_sol) / lambda_val

        return torch.from_numpy(gradient).to(device)


if __name__ == '__main__':
    m = EdgeMatching_autograd()
    input = torch.rand(9, requires_grad=True)
    print(input.reshape(-1, 3))
    out = m.apply(input)
    print(out.reshape(-1, 3))
