from ortools.linear_solver import pywraplp


def solve_matrix_for_row_size_two(matrix):
    """
    Compute the value and the Row player strategy.
    The size of the matrix is 2*2.
    x (=(x1, x2))is the strategy of the Row player.
    a_i,j is the value of the matrix.
    Maximize 'L' such that
    a_1,1 * x1 + a_2,1 * x2 >= L
    a_1,2 * x1 + a_2,2 * x2 >= L
    x1 + x2 = 1
    xi >= 0
    """
    print(f"start solving matrix for row. {matrix}")
    # Instantiate a solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create variables and let them take on any non-negative value.
    L = solver.NumVar(0, solver.infinity(), 'L')
    x1 = solver.NumVar(0, 1, 'x1')
    x2 = solver.NumVar(0, 1, 'x2')

    # Constraints
    # a_1,1 * x1 + a_2,1 * x2 - L >= 0
    constraint1 = solver.Constraint(0, solver.infinity())
    constraint1.SetCoefficient(x1, matrix[0][0])
    constraint1.SetCoefficient(x2, matrix[1][0])
    constraint1.SetCoefficient(L, -1)
    # a_1,2 * x1 + a_2,2 * x2 - L >= 0
    constraint2 = solver.Constraint(0, solver.infinity())
    constraint2.SetCoefficient(x1, matrix[0][1])
    constraint2.SetCoefficient(x2, matrix[1][1])
    constraint2.SetCoefficient(L, -1)
    # x1 + x2 = 1
    constraint3 = solver.Constraint(1, 1)
    constraint3.SetCoefficient(x1, 1)
    constraint3.SetCoefficient(x2, 1)

    # Objective function, Maximize L.
    objective = solver.Objective()
    objective.SetCoefficient(L, 1)
    objective.SetMaximization()

    solver.Solve()

    print(f'value = {L.solution_value()}')
    print('strategy')
    print(f'x1 = {x1.solution_value()}')
    print(f'x2 = {x2.solution_value()}')
    print()


def solve_matrix_for_row(matrix):
    """
    Compute the value and the Row player strategy.
    The size of the matrix is n*m.
    x (=(x1, x2))is the strategy of the Row player.
    a_i,j is the value of the matrix.
    Maximize 'L' such that
    a_1,i * x1 + a_2,i * x2 + ...  >= L  # * m
    x1 + x2 + xi ... = 1
    xi >= 0

    return the optimal strategy of row
    """
    print(f"start solving matrix for row. {matrix}")
    matrix, min_value = positive_matrix(matrix)  # if negative number in matrix
    n = len(matrix)
    m = len(matrix[0])
    # Instantiate a solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create variables and let them take on any non-negative value.
    L = solver.NumVar(0, solver.infinity(), 'L')
    variables = [solver.NumVar(0, 1, 'x'+str(i)) for i in range(n)]

    # Constraints
    # a_1,i * x1 + a_2,i * x2 - L >= 0
    for c in range(m):
        constraint = solver.Constraint(0, solver.infinity())
        for r in range(n):
            constraint.SetCoefficient(variables[r], matrix[r][c])
        constraint.SetCoefficient(L, -1)
    # x1 + x2 + xi ... = 1
    constraint3 = solver.Constraint(1, 1)
    for v in variables:
        constraint3.SetCoefficient(v, 1)

    # Objective function, Maximize L.
    objective = solver.Objective()
    objective.SetCoefficient(L, 1)  # maximize L*1
    objective.SetMaximization()

    solver.Solve()

    print(f'value = {L.solution_value() + min_value}')
    print('strategy')
    strategy = []
    for i, v in enumerate(variables):
        print(f'x{i} = {v.solution_value()}')
        strategy.append(v.solution_value())
    print()
    return strategy


def solve_matrix_for_column(matrix):
    """
    Compute the value and the Column player strategy.
    The size of the matrix is n*m.
    y (=(y1, y2, ...))is the strategy of the Column player.
    a_i,j is the value of the matrix.
    Minimize 'M' such that
    a_i,1 * y1 + a_i,2 * y2 + ...  <= M  # * n
    y1 + y2 + yi ... = 1
    yi >= 0

    return the optimal strategy of row
    """
    print(f"start solving matrix for column. {matrix}")
    matrix, min_value = positive_matrix(matrix)
    n = len(matrix)
    m = len(matrix[0])
    # Instantiate a solver
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Create variables and let them take on any non-negative value.
    M = solver.NumVar(0, solver.infinity(), 'M')
    variables = [solver.NumVar(0, 1, 'y'+str(i)) for i in range(m)]

    # Constraints
    # a_i,1 * y1 + a_i,2 * y2 - M <= 0
    for r in range(n):
        constraint = solver.Constraint(-solver.infinity(), 0)
        for c in range(m):
            constraint.SetCoefficient(variables[c], matrix[r][c])
        constraint.SetCoefficient(M, -1)
    # y1 + y2 + yi ... = 1
    constraint3 = solver.Constraint(1, 1)
    for v in variables:
        constraint3.SetCoefficient(v, 1)

    # Objective function, Maximize L.
    objective = solver.Objective()
    objective.SetCoefficient(M, 1)  # minimize M*1
    objective.SetMinimization()

    solver.Solve()

    print(f'value = {M.solution_value() + min_value}')
    print('strategy')
    strategy = []
    for i, v in enumerate(variables):
        print(f'x{i} = {v.solution_value()}')
        strategy.append(v.solution_value())
    print()
    return strategy


def all_row_values(matrix, column_strategy):
    """
    Compute all values for all row pure strategy
    """
    print("value for each row pure strategy")
    for row in matrix:
        value = 0
        for a, c in zip(row, column_strategy):
            value += a * c
        print(f'value = {value}')
    print()


def all_column_values(matrix, row_strategy):
    """
    Compute all values for all column pure strategy
    """
    print("value for each column pure strategy")
    for ci in range(len(matrix[0])):
        value = 0
        for ri in range(len(matrix)):
            value += matrix[ri][ci] * row_strategy[ri]
        print(f'value = {value}')
    print()


def positive_matrix(matrix):
    """
    All values should be positive.
    When negative number in matrix, add -min_value to all values
    """
    min_value = 0
    for row in matrix:
        for v in row:
            min_value = min(min_value, v)
    if min_value >= 0:
        return matrix
    positive = []
    for row in matrix:
        positive.append([v - min_value for v in row])
    return positive, min_value


if __name__ == '__main__':
    # game_matrix = [[1, 4], [3, 2]]
    # solve_matrix_for_row_size_two(game_matrix)
    # game_matrix = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    game_matrix = [[-1, 1], [0, -1]]
    # game_matrix = [[0, 2], [1, 0]]
    r_strategy = solve_matrix_for_row(game_matrix)
    c_strategy = solve_matrix_for_column(game_matrix)
    all_row_values(game_matrix, c_strategy)
    all_column_values(game_matrix, r_strategy)
