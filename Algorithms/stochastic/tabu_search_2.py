import random

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker
from Praca.problem_setup import objective_function, problem_configuration


def random_vector(search_space):
    return [random.uniform(min_val, max_val) for min_val, max_val in search_space]

def local_search(best, tabu_list, search_space, max_no_improv):
    count = 0
    while count < max_no_improv:
        candidate = random_vector(search_space)
        if candidate not in tabu_list and (best is None or objective_function(candidate) < objective_function(best)):
            best = candidate
            count = 0
        else:
            count += 1
        tabu_list.append(candidate)
        if len(tabu_list) > max_tabu_size:
            tabu_list.pop(0)
    return best

@time_tracker
@memory_tracker
def search(search_space, max_iter, max_no_improv, max_tabu_size):
    best = {}
    best_cost = None
    tabu_list = []
    for _ in range(max_iter):
        candidate = random_vector(search_space)
        candidate = local_search(candidate, tabu_list, search_space, max_no_improv)
        if best_cost is None or objective_function(candidate) < objective_function(best_cost):
            best_cost = candidate
        
        print(f"Iteration: Arguments = {best_cost}, Objective Function Value = {objective_function(best_cost)}")

    best['cost'] = objective_function(best_cost)
    best['vector'] = best_cost

    return best


if __name__ == '__main__':

    algorithm_name = 'Przeszukiwanie z Zakazem'
    
    # problem configuration
    problem_size, search_space, optimal_solution = problem_configuration()
    # algorithm configuration
    max_iter = 100
    max_no_improv = 50
    max_tabu_size = 10

    # execute the algorithm

    for i in range(100):

        best = search(search_space, max_iter, max_no_improv, max_tabu_size)
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
