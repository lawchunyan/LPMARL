import numpy as np
import torch
from ortools.linear_solver import pywraplp


def solve_LP(obj_coeff, constraint_coeff=None):
    num_ag = obj_coeff.shape[0]
    num_task = obj_coeff.shape[1]

    solver = pywraplp.Solver.CreateSolver('GLOP')
    # solver = pywraplp.Solver('SolveStigler', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # set objective
    objective = solver.Objective()

    # create empty matrix of the decision variables
    x = [[[] for _ in range(num_task)] for _ in range(num_ag)]

    # set decision variable
    for i in range(num_ag):
        for j in range(num_task):
            x[i][j] = solver.NumVar(0, 1, 'x_{}{}'.format(i, j))

            # objective coefficient
            objective.SetCoefficient(x[i][j], obj_coeff[i][j].item())

    objective.SetMaximization()

    constraints_ag = [0] * num_ag
    # Set constraint
    for i in range(num_ag):
        constraints_ag[i] = solver.Constraint(1.0, 1.0)
        for j in range(num_task):
            # constraints[i].SetCoefficient(x[i][j], constraint_coeff[i][j])
            constraints_ag[i].SetCoefficient(x[i][j], 1)

    constraints_task = [0] * num_task
    for t in range(num_task):
        constraints_task[t] = solver.Constraint(-solver.infinity(), 4.0)
        for ag in range(num_ag):
            constraints_task[t].SetCoefficient(x[ag][t], 1)

    status = solver.Solve()
    if status == solver.OPTIMAL:
        A = [[x[i][j].solution_value() for j in range(num_task)] for i in range(num_ag)]
        A = torch.Tensor(A)
        return A, [c.dual_value() for c in constraints_ag], [c.dual_value() for c in constraints_task]


if __name__ == '__main__':
    num_ag = 7
    num_task = 5
    obj_coeff = np.zeros((num_ag, num_task))
    # constraint_coeff = np.random.random((num_ag, num_task))

    A = solve_LP(obj_coeff)
    print(A)

