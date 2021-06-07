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


class OptLayer(nn.Module):
    def __init__(self, variables, parameters, objective, inequalities, equalities, **cvxpy_opts):
        super().__init__()
        self.variables = variables
        self.parameters = parameters
        self.objective = objective
        self.inequalities = inequalities
        self.equalities = equalities
        self.cvxpy_opts = cvxpy_opts

        # create the cvxpy problem with objective, inequalities, equalities
        self.cp_inequalities = [ineq(*variables, *parameters) <= 0 for ineq in inequalities]
        self.cp_equalities = [eq(*variables, *parameters) == 0 for eq in equalities]
        self.problem = cp.Problem(cp.Minimize(objective(*variables, *parameters)),
                                  self.cp_inequalities + self.cp_equalities)

    def forward(self, *batch_params):
        out, J = [], []
        # solve over minibatch by just iterating
        for batch in range(len(batch_params[0])):
            # solve the optimization problem and extract solution + dual variables
            params = [p[batch] for p in batch_params]
            with torch.no_grad():
                for i, p in enumerate(self.parameters):
                    p.value = params[i].cpu().double().numpy()
                self.problem.solve(**self.cvxpy_opts)
                z = [torch.tensor(v.value).type_as(params[0]) for v in self.variables]
                lam = [torch.tensor(c.dual_value).type_as(params[0]) for c in self.cp_inequalities]
                nu = [torch.tensor(c.dual_value).type_as(params[0]) for c in self.cp_equalities]

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
                g = [ineq(*z, *params) for ineq in self.inequalities]
                dnu = [eq(*z, *params) for eq in self.equalities]
                L = (self.objective(*z, *params) +
                     sum((u * v).sum() for u, v in zip(lam, g)) + sum((u * v).sum() for u, v in zip(nu, dnu)))
                dz = autograd.grad(L, z, create_graph=True)
                dlam = [lam[i] * g[i] for i in range(len(lam))]
                return dz, dlam, dnu

            # compute residuals and re-engage autograd tape
            y = vec(z, lam, nu)
            y = y - vec(*kkt([z_.clone().detach().requires_grad_() for z_ in z], lam, nu, *params))
            # vec(*kkt(*mat(y), *params))

            # compute jacobian and backward hook
            J.append(autograd.functional.jacobian(lambda x: vec(*kkt(*mat(x), *params)), y))

            def return_grad(grad, b=batch):
                out = torch.mm(torch.inverse(J[b].transpose(0, 1)), grad[:, None])[:, 0]
                return out

            # y.register_hook(lambda grad, b=batch: torch.solve(grad[:, None], J[b].transpose(0, 1))[0][:, 0])
            y.register_hook(return_grad)

            out.append(mat(y)[0])
        out = [torch.stack(o, dim=0) for o in zip(*out)]
        # from optim.lp_solver import solve_LP
        # sol1 = solve_LP(batch_params[0][0].reshape(8, 8))[0].detach().numpy()
        # coeff = batch_params[0][0].reshape(2, 2).detach().numpy()
        # sol2 = dn([torch.tensor(v.value).type_as(params[0]) for v in self.variables][0].reshape(2, 2))
        # O = dn(out[0].reshape(2, 2))
        # val2 = self.problem.value
        return out[0] if len(out) == 1 else tuple(out)


class MatchingLayer(nn.Module):
    def __init__(self, n, m, coeff, device='cpu'):
        """
        :param n: num_ag
        :param m: num_target
        """
        super().__init__()
        self.device = device

        self.x = cp.Variable(n * m)
        self.coeff = cp.Parameter(n * m)
        self.A = cp.Parameter((n, n * m))
        self.b = cp.Parameter(n)
        self.C = cp.Parameter((m, n * m))
        self.d = cp.Parameter(m)
        variables = [self.x]
        self.parameters = [self.coeff, self.A, self.b, self.C, self.d]
        # cp_ineq = [inequality(*variables, *self.parameters) <= 0] + [0 <= self.x, self.x <= 1]
        # cp_eq = [equality(*variables, *self.parameters) == 0]

        self.optimLayer = OptLayer(variables=variables, parameters=self.parameters, objective=objective,
                                   inequalities=[inequality, ineq_lb, ineq_up], equalities=[equality])
        A_val = torch.zeros((n, n * m)).to(self.device)
        for i in range(n):
            A_val[i, i * m:(i + 1) * m] = 1
        self.A_val = A_val
        self.b_val = torch.ones(n).to(self.device)

        # set C value
        temp = torch.eye(m).to(self.device)
        C_value = torch.cat([temp for _ in range(n)], axis=-1)
        self.C_val = C_value
        self.d_val = torch.ones(m).to(self.device) * coeff

    def forward(self, lst_coeff):
        n_batch = len(lst_coeff)
        A_batch = [self.A_val] * n_batch
        b_batch = [self.b_val] * n_batch
        C_batch = [self.C_val] * n_batch
        d_batch = [self.d_val] * n_batch

        sol = self.optimLayer(lst_coeff, A_batch, b_batch, C_batch, d_batch)
        return sol


def objective(x, coeff, A, b, C, d):
    # return - coeff @ x + 1e-5 * cp.sum_squares(x)
    # return - coeff @ x + 1e-5 * x @ x
    return - coeff @ x + 1e-5 * sum(x * x)  # ok


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
    x = cp.Variable(2)
    obj = cp.Minimize(x[0] + cp.norm(x, 1) + cp.sum_squares(x))
    obj = cp.Minimize(cp.norm(x, 2))
    constraints = [x >= 2]
    prob = cp.Problem(obj, constraints)

    # Solve with OSQP.
    prob.solve(solver=cp.OSQP)
