import random

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker

def objective_function(vector):
    return sum(x ** 2.0 for x in vector)

def random_vector(minmax):
    return [minmax[i][0] + ((minmax[i][1] - minmax[i][0]) * random.random()) for i in range(len(minmax))]

def create_particle(search_space, vel_space):
    particle = {}
    particle['position'] = random_vector(search_space)
    particle['cost'] = objective_function(particle['position'])
    particle['b_position'] = list(particle['position'])
    particle['b_cost'] = particle['cost']
    particle['velocity'] = random_vector(vel_space)
    return particle

def get_global_best(population, current_best=None):
    population.sort(key=lambda x: x['cost'])
    best = population[0]
    if current_best is None or best['cost'] <= current_best['cost']:
        current_best = {}
        current_best['position'] = list(best['position'])
        current_best['cost'] = best['cost']
    return current_best

def update_velocity(particle, gbest, max_v, c1, c2):
    for i, v in enumerate(particle['velocity']):
        v1 = c1 * random.random() * (particle['b_position'][i] - particle['position'][i])
        v2 = c2 * random.random() * (gbest['position'][i] - particle['position'][i])
        particle['velocity'][i] = v + v1 + v2
        particle['velocity'][i] = max_v if particle['velocity'][i] > max_v else particle['velocity'][i]
        particle['velocity'][i] = -max_v if particle['velocity'][i] < -max_v else particle['velocity'][i]

def update_position(particle, bounds):
    for i, v in enumerate(particle['position']):
        particle['position'][i] = v + particle['velocity'][i]
        if particle['position'][i] > bounds[i][1]:
            particle['position'][i] = bounds[i][1] - abs(particle['position'][i] - bounds[i][1])
            particle['velocity'][i] *= -1.0
        elif particle['position'][i] < bounds[i][0]:
            particle['position'][i] = bounds[i][0] + abs(particle['position'][i] - bounds[i][0])
            particle['velocity'][i] *= -1.0

def update_best_position(particle):
    if particle['cost'] <= particle['b_cost']:
        particle['b_cost'] = particle['cost']
        particle['b_position'] = list(particle['position'])

@time_tracker
@memory_tracker
def search(max_gens, search_space, vel_space, pop_size, max_vel, c1, c2):

    pop = [create_particle(search_space, vel_space) for _ in range(pop_size)]
    best = get_global_best(pop)
    for gen in range(max_gens):
        for particle in pop:
            update_velocity(particle, best, max_vel, c1, c2)
            update_position(particle, search_space)
            particle['cost'] = objective_function(particle['position'])
            update_best_position(particle)
        best = get_global_best(pop, best)
        print(f" > gen {gen+1}, fitness={best['cost']}")

    return best

if __name__ == "__main__":
    
    algorithm_name = 'particle swarm optimization algorithm'
    optimal_solution = 0
    # problem configuration
    problem_size = 3
    search_space = [[-10, 10] for _ in range(problem_size)]
    # algorithm configuration
    vel_space = [[-1, 1] for _ in range(problem_size)]
    max_gens = 100
    pop_size = 50
    max_vel = 100.0
    c1, c2 = 2.0, 2.0
    # execute the algorithm
    for i in range(100):

        best = search(max_gens, search_space, vel_space, pop_size, max_vel, c1,c2)
        solution = best['cost']
        error = abs(optimal_solution - solution)
        arguments = best['position']
        total_time = best['time']
        total_memory = best['memory']

        print("Done. Best Solution: c={}, v={}".format(solution, arguments))
        

        csv_file_name = 'DATA.csv'
        data = [algorithm_name, solution, error ,arguments, total_time, total_memory]

        write_to_csv(csv_file_name, data)

    calculate_average(csv_file_name,'method', algorithm_name, [ 'function_value','error','time_duration', 'total_memory'])
