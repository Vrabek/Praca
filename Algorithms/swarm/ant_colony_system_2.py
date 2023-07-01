#zaimplementowane na podstawie pracy Clever Algorithms Nature-Inspired Programming Recipes
import random

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker
from Praca.problem_setup import objective_function, problem_configuration

def random_vector(search_space):
    return [random.uniform(search_space[i][0], search_space[i][1]) for i in range(len(search_space))]

def initialise_pheromone_matrix(problem_size, init_pher):
    return [[init_pher for _ in range(problem_size)] for _ in range(problem_size)]

def calculate_choices(last_city, exclude, pheromone, heuristic, c_heur, c_hist):
    choices = []
    for i in range(len(heuristic)):
        if i in exclude:
            continue
        prob = {'index': i}
        prob['history'] = pheromone[last_city][i] ** c_hist
        prob['heuristic'] = heuristic[i] ** c_heur
        prob['prob'] = prob['history'] * prob['heuristic']
        choices.append(prob)
    return choices

def prob_select(choices):
    total = sum(choice['prob'] for choice in choices)
    if total == 0.0:
        return random.choice(choices)['index']
    r = random.uniform(0, total)
    prob_sum = 0.0
    for choice in choices:
        prob_sum += choice['prob']
        if prob_sum >= r:
            return choice['index']
    return choices[-1]['index']

def greedy_select(choices):
    return max(choices, key=lambda x: x['prob'])['index']

def global_update_pheromone(pheromone, cand, decay):
    for i in range(len(cand['vector'])):
        x = int(cand['vector'][i]) % len(pheromone)  # Wrap the index within the range of the pheromone matrix
        y = int(cand['vector'][0]) % len(pheromone) if i == len(cand['vector']) - 1 else int(cand['vector'][i + 1]) % len(pheromone)
        value = ((1.0 - decay) * pheromone[x][y]) + (decay / cand['cost'])
        pheromone[x][y] = value
        pheromone[y][x] = value

def local_update_pheromone(pheromone, cand, c_local_phero, init_phero):
    for i in range(len(cand['vector'])):
        x = int(cand['vector'][i])
        y = int(cand['vector'][0]) if i == len(cand['vector']) - 1 else int(cand['vector'][i + 1])
        value = ((1.0 - c_local_phero) * pheromone[x][y]) + (c_local_phero * init_phero)
        pheromone[x][y] = value
        pheromone[y][x] = value

@time_tracker
@memory_tracker
def search(problem_size, search_space, max_iterations, num_ants, decay, c_heur, c_local_phero, c_greed):
    
    best = {'vector': random_vector(search_space)}
    best['cost'] = objective_function(best['vector'])
    init_pheromone = 1.0 / (problem_size * 1.0 * best['cost'])
    pheromone = initialise_pheromone_matrix(problem_size, init_pheromone)
    for iteration in range(max_iterations):
        for _ in range(num_ants):
            cand = {}
            cand['vector'] = random_vector(search_space)
            cand['cost'] = objective_function(cand['vector'])
            if cand['cost'] < best['cost']:
                best = cand
        global_update_pheromone(pheromone, best, decay)
        print(" > iteration {}, best argument={}, best value={}".format(iteration + 1, best['vector'], best['cost']))

    return best

if __name__ == "__main__":
    
    algorithm_name = 'System Kolonii Mrowek'
    
    # problem configuration
    problem_size, search_space, optimal_solution = problem_configuration()
    # algorithm configuration
    max_iterations = 100
    num_ants = 10
    decay = 0.1
    c_heur = 2.5
    c_local_phero = 0.1
    c_greed = 0.9
    # execute the algorithm
    
    for i in range(100):

        best = search(problem_size, search_space, max_iterations, num_ants, decay, c_heur, c_local_phero, c_greed)
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
