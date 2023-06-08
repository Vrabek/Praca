def objective_function(vector):
    return sum(x_i ** 2.0 for x_i in vector)


def problem_configuration():

    problem_size = 3
    search_space = [[-10, +10] for _ in range(problem_size)]
    optimal_solution = 0

    return problem_size, search_space, optimal_solution
