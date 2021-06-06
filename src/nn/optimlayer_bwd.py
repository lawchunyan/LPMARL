import cvxpy as cp
import torch
import torch.nn as nn
import torch.autograd as autograd
from itertools import accumulate

default_solving_dict = {
    'max_iters': 100,
    'abstol': 1e-8,
    'reltol': 1e-8,
    'feastol': 1e-8, }


class OptLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, problem, objective, variables, inequalities, equalities, cp_inequalities, cp_equalities,
                parameters, *batch_params):
        out, J = [], []
        for b in range(len(batch_params[0])):
            params = [p[b] for p in batch_params]
            with torch.no_grad():
                for i, p in enumerate(parameters):
                    p.value = params[i].double().numpy()
                problem.solve()
                z = [torch.tensor(v.value).type_as(params[0]) for v in variables]
                lam = [torch.tensor(c.dual_value).type_as(params[0]) for c in cp_inequalities]
                nu = [torch.tensor(c.dual_value).type_as(params[0]) for c in cp_equalities]

            # convenience routines to "flatten" and "unflatten" (z,lam,nu)
            def vec(z, lam, nu):
                return torch.cat([a.view(-1) for b in [z, lam, nu] for a in b])

            def mat(x):
                sz = [0] + list(accumulate([a.numel() for b in [z, lam, nu] for a in b]))
                val = [x[a:b] for a, b in zip(sz, sz[1:])]
                return ([val[i].view_as(z[i]) for i in range(len(z))],
                        [val[i + len(z)].view_as(lam[i]) for i in range(len(lam))],
                        [val[i + len(z) + len(lam)].view_as(nu[i]) for i in range(len(nu))])

            # computes the KKT residual
            def kkt(z, lam, nu, *params):
                g = [ineq(*z, *params) for ineq in inequalities]
                dnu = [eq(*z, *params) for eq in equalities]
                # print(dnu)
                L = (objective(*z, *params) +
                     sum((u * v).sum() for u, v in zip(lam, g)) + sum((u * v).sum() for u, v in zip(nu, dnu)))
                dz = autograd.grad(L, z, create_graph=True)
                dlam = [lam[i] * g[i] for i in range(len(lam))]
                return dz, dlam, dnu

            y = vec(z, lam, nu)

            ctx.lam = lam
            ctx.nu = nu

            y = y - vec(*kkt([z_.clone().detach().requires_grad_() for z_ in z], lam, nu, *params))

        return y[0]

    @staticmethod
    def backward(ctx, grad_output):

        return


class MatchingLayer(nn.Module):
    def __init__(self, n, m):
        """
        :param n: num_ag
        :param m: num_target
        """
        super().__init__()
        x = cp.Variable(n * m)
        self.variables = [x]

        coeff = cp.Parameter(n * m)
        self.A = cp.Parameter((n, n * m))
        self.b = cp.Parameter(n)
        self.C = cp.Parameter((m, n * m))
        self.d = cp.Parameter(m)

        self.parameters = [coeff, self.A, self.b, self.C, self.d]

        self.objective = objective
        self.inequalities = [inequality, ineq_lb, ineq_up]
        self.equalities = [equality]

        # cvxpy problem
        self.cp_inequalities = [ineq(*self.variables, *self.parameters) <= 0 for ineq in self.inequalities]
        self.cp_equalities = [eq(*self.variables, *self.parameters) == 0 for eq in self.equalities]
        self.problem = cp.Problem(cp.Minimize(self.objective(*self.variables, *self.parameters)),
                                  self.cp_equalities + self.cp_inequalities)

        # self.optimLayer = OptLayer(variables=variables, parameters=self.parameters, objective=objective,
        #                            inequalities=[inequality, ineq_lb, ineq_up], equalities=[equality])

        self.layer = OptLayer()

        A_val = torch.zeros((n, n * m))
        for i in range(n):
            A_val[i, i * m:(i + 1) * m] = 1
        self.A_val = A_val
        self.b_val = torch.ones(n)

        # set C value
        temp = torch.eye(m)
        C_value = torch.cat([temp for _ in range(n)], axis=-1)
        self.C_val = C_value
        self.d_val = torch.ones(m) * 4

    def forward(self, lst_coeff):
        n_batch = len(lst_coeff)
        A_batch = [self.A_val] * n_batch
        b_batch = [self.b_val] * n_batch
        C_batch = [self.C_val] * n_batch
        d_batch = [self.d_val] * n_batch

        self.set_parameters(lst_coeff, A_batch, b_batch, C_batch, d_batch)

        # sol = self.solve_batched_problem(lst_coeff, A_batch, b_batch, C_batch, d_batch)
        sol = self.layer.apply(self.problem, self.objective, self.variables, self.inequalities, self.equalities,
                               self.cp_inequalities, self.cp_equalities, self.parameters,
                               lst_coeff, A_batch, b_batch, C_batch, d_batch)

        return sol

    def set_parameters(self, *batch_params):
        num_batch = len(batch_params[0])
        for b in range(num_batch):
            params = [p[b] for p in batch_params]
            with torch.no_grad():
                for i, p in enumerate(self.parameters):
                    p.value = params[i].double().numpy()


def objective(x, coeff, A, b, C, d):
    return - coeff @ x + 1e-5 * x @ x


def equality(x, coeff, A, b, C, d):
    return A @ x - b


# def equality2(x, coeff, A, b, C, d):
#     return x * (1 - x)


def inequality(x, coeff, A, b, C, d):
    return C @ x - d


# def eqality
#
def ineq_lb(x, coeff, A, b, C, d):
    return - x


def ineq_up(x, coeff, A, b, C, d):
    return x - 1


if __name__ == '__main__':
    pass
