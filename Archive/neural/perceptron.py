import random

def objective_function(vector):
    return sum(x ** 2.0 for x in vector)

def random_vector(minmax):
    return [minmax[i][0] + ((minmax[i][1] - minmax[i][0]) * random.random()) for i in range(len(minmax))]

def initialize_weights(problem_size):
    minmax = [[-1.0, 1.0] for _ in range(problem_size)]
    return random_vector(minmax)

def get_gradient(vector):
    return [2.0 * x for x in vector]

def update_weights(weights, gradient, learning_rate):
    return [weights[i] - learning_rate * gradient[i] for i in range(len(weights))]

def train_weights(weights, iterations, learning_rate):
    for epoch in range(iterations):
        gradient = get_gradient(weights)
        weights = update_weights(weights, gradient, learning_rate)
        error = objective_function(weights)
        print(f"> epoch={epoch}, error={error}")
    return weights

def execute(problem_size, iterations, learning_rate):  
    weights = initialize_weights(problem_size)
    weights = train_weights(weights, iterations, learning_rate)
    return weights

if __name__ == "__main__":
    # problem configuration
    problem_size = 2  # number of variables in the problem

    # algorithm configuration
    iterations = 10000
    learning_rate = 0.1  

    # execute the algorithm
    weights = execute(problem_size, iterations, learning_rate)
    print("Final weights: ", weights)
    print("Final objective function value: ", objective_function(weights))
