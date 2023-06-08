import random

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker
from Praca.problem_setup import objective_function

def rand_in_bounds(min_val, max_val):
    return min_val + ((max_val - min_val) * random.random())

def random_vector(search_space):
    return [rand_in_bounds(search_space[i][0], search_space[i][1]) for i in range(len(search_space))]

def initialize_particle(search_space):
    particle = {}
    particle['position'] = random_vector(search_space)
    particle['velocity'] = [0.0] * len(search_space)
    particle['best_position'] = list(particle['position'])
    particle['best_value'] = float('inf')
    return particle

def update_velocity(particle, global_best_position, c1, c2):
    for i in range(len(particle['velocity'])):
        r1 = random.random()
        r2 = random.random()
        cognitive_velocity = c1 * r1 * (particle['best_position'][i] - particle['position'][i])
        social_velocity = c2 * r2 * (global_best_position[i] - particle['position'][i])
        particle['velocity'][i] = particle['velocity'][i] + cognitive_velocity + social_velocity

def update_position(particle, search_space):
    for i in range(len(particle['position'])):
        particle['position'][i] = particle['position'][i] + particle['velocity'][i]
        # Check if the new position is within the search space bounds
        particle['position'][i] = max(particle['position'][i], search_space[i][0])
        particle['position'][i] = min(particle['position'][i], search_space[i][1])


@time_tracker
@memory_tracker
def optimize(search_space, max_iter, swarm_size, c1, c2):
    best = {}
    global_best_position = None
    global_best_value = float('inf')
    swarm = [initialize_particle(search_space) for _ in range(swarm_size)]
    
    for iteration in range(max_iter):
        for particle in swarm:
            particle_value = objective_function(particle['position'])
            
            if particle_value < particle['best_value']:
                particle['best_position'] = list(particle['position'])
                particle['best_value'] = particle_value
            
            if particle_value < global_best_value:
                global_best_position = list(particle['position'])
                global_best_value = particle_value
        
            update_velocity(particle, global_best_position, c1, c2)
            update_position(particle, search_space)
        
        print("Iteration {}: Best Value = {}".format(iteration + 1, global_best_value))
        best['cost'] = global_best_value
        best['vector'] = global_best_position

    return best


if __name__ == "__main__":

    algorithm_name = 'dendritic cell algorithm'
    optimal_solution = 0
    # problem configuration
    problem_size = 3
    search_space = [[-10, +10] for _ in range(problem_size)]
    # algorithm configuration
    max_iter = 100
    swarm_size = 20
    c1 = 2.0  # Cognitive parameter
    c2 = 2.0  # Social parameter
    # execute the algorithm
    for i in range(100):

        best = optimize(search_space, max_iter, swarm_size, c1, c2)
        solution = best['cost']
        error = abs(optimal_solution - solution)
        arguments = best['vector']
        total_time = best['time']
        total_memory = best['memory']

        print("Done. Best Solution: c={}, v={}".format(solution, arguments))
        
        csv_file_name = 'DATA.csv'
        data = [algorithm_name, solution, error ,arguments, total_time, total_memory]

        write_to_csv(csv_file_name, data)

    calculate_average(csv_file_name,'method', algorithm_name, [ 'function_value','error','time_duration', 'total_memory'])
