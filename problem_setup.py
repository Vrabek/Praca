def objective_function(vector):
    #f(x, y) = x^2 * (x - 2)^2 + y^2 * (y - 2)^2
    return sum(x_i ** 2.0 for x_i in vector)

#def objective_function(vector):
#    x, y = vector
#    return x**2 * (x - 2)**2 + y**2 * (y - 2)**2

def problem_configuration():

    problem_size = 2
    search_space = [[-10, +10] for _ in range(problem_size)]
    optimal_solution = 0

    return problem_size, search_space, optimal_solution
