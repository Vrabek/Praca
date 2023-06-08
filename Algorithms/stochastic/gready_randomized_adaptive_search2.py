import random

from Praca.decorators import memory_tracker, time_tracker
from Praca.csv_utils import write_to_csv, calculate_average
from Praca.problem_setup import objective_function, problem_configuration


def random_vector(search_space):
    return [random.uniform(min_val, max_val) for min_val, max_val in search_space]

def construct_greedy_solution(search_space, alpha):
    candidate = {}
    candidate['vector'] = random_vector(search_space)
    candidate['cost'] = objective_function(candidate['vector'])
    return candidate

def local_search(best, max_no_improv):
    count = 0
    while count < max_no_improv:
        candidate = construct_greedy_solution(search_space, alpha=0)
        candidate['cost'] = objective_function(candidate['vector'])
        if candidate['cost'] < best['cost']:
            best = candidate
            count = 0
        else:
            count += 1
    return best

@time_tracker
@memory_tracker
def search(search_space, problem_size, max_iter, max_no_improv):
    best = None
    for iter in range(max_iter):
        candidate = construct_greedy_solution(search_space, alpha=0.3)
        candidate = local_search(candidate, max_no_improv)
        if best is None or candidate['cost'] < best['cost']:
            best = candidate
        print(f"Iteration {iter + 1}: Arguments = {best['vector']}, Objective Function Value = {best['cost']}")

    return best

if __name__ == '__main__':

    algorithm_name = 'greedy randomized adaptive search'

    # problem configuration
    problem_size, search_space, optimal_solution = problem_configuration()
    # algorithm configuration
    max_iter = 100
    max_no_improv = 50
    # execute the algorithm
    for i in range(100):

        best = search(search_space, max_iter, max_iter, max_no_improv)
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
