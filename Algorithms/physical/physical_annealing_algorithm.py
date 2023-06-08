import random
import math

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker
from Praca.problem_setup import objective_function


def random_vector(search_space):
    return [random.uniform(search_space[i][0], search_space[i][1]) for i in range(len(search_space))]

def create_neighbor(current, search_space):
    candidate = {'vector': list(current['vector'])}
    index = random.randint(0, len(candidate['vector']) - 1)
    candidate['vector'][index] = random.uniform(search_space[index][0], search_space[index][1])
    candidate['cost'] = objective_function(candidate['vector'])
    return candidate

def should_accept(candidate, current, temp):
    if candidate['cost'] <= current['cost']:
        return True
    return math.exp((current['cost'] - candidate['cost']) / temp) > random.random()

@time_tracker
@memory_tracker
def search(search_space, max_iter, max_temp, temp_change):

    current = {'vector': random_vector(search_space)}
    current['cost'] = objective_function(current['vector'])
    temp, best = max_temp, current
    for iteration in range(max_iter):
        candidate = create_neighbor(current, search_space)
        temp *= temp_change
        if should_accept(candidate, current, temp):
            current = candidate
        if candidate['cost'] < best['cost']:
            best = candidate
        
        print(f"> iteration {iteration+1}, temp={temp}, best={best['cost']}")

    return best

if __name__ == "__main__":
    
    algorithm_name = 'physical annealing algorithm'
    optimal_solution = 0
    # problem configuration
    problem_size =  3
    search_space = [[-10, +10] for _ in range(problem_size)]
    # algorithm configuration
    max_iter = 100
    max_temp = 100000.0
    temp_change = 0.98
    # execute the algorithm

    for i in range(100):

        best = search(search_space, max_iter, max_temp, temp_change)
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
