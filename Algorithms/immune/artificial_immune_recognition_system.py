import random
import math

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker

# objective function
def objective_function(vector):
    return sum(x ** 2.0 for x in vector)

def random_vector(n_dimensions, search_space):
    return [random.uniform(search_space[i][0], search_space[i][1]) for i in range(n_dimensions)]

def create_cell(vector):
    return {'vector': vector, 'score': objective_function(vector)}

def initialize_cells(n_cells, n_dimensions, search_space):
    return [create_cell(random_vector(n_dimensions, search_space)) for _ in range(n_cells)]

def distance(c1, c2):
    return math.sqrt(sum((c1[i]-c2[i])**2 for i in range(len(c1))))

def mutate_cell(cell, best_match, range_val, search_space):
    cell['vector'] = [min(max(v + (random.uniform(-1,1) * range_val), search_space[i][0]), search_space[i][1]) for i, v in enumerate(cell['vector'])]
    cell['score'] = objective_function(cell['vector'])
    return cell

def create_arb_pool(best_match, clone_rate, mutate_rate, search_space):
    pool = [best_match]
    num_clones = round(clone_rate * mutate_rate)
    for _ in range(num_clones):
        cell = create_cell(best_match['vector'].copy())
        pool.append(mutate_cell(cell, best_match, 1.0 - best_match['score'], search_space))
    return pool

def competition_for_resources(pool, clone_rate, max_res):
    for cell in pool:
        cell['resources'] = clone_rate / cell['score']  # We want to minimize the function, so we divide instead of multiply
    pool.sort(key=lambda x: x['resources'], reverse=True)  # Highest resources first
    total_resources = sum(cell['resources'] for cell in pool)
    while total_resources > max_res:
        cell = pool.pop()  # Remove the cell with the smallest resources
        total_resources -= cell['resources']

def train_system(mem_cells, num_patterns, clone_rate, mutate_rate, max_res, search_space):
    for i in range(num_patterns):
        best_match = min(mem_cells, key=lambda x: x['score'])
        print(f"Iteration {i+1}, Best score: {best_match['score']}, Best vector: {best_match['vector']}")
        pool = create_arb_pool(best_match, clone_rate, mutate_rate, search_space)
        competition_for_resources(pool, clone_rate, max_res)
        mem_cells.extend(pool)  # Add all cells from the pool to the memory cells
        mem_cells.sort(key=lambda x: x['score'])  # Sort the cells by their score
        if len(mem_cells) > max_res:  # If we have too many cells, remove the worst ones
            mem_cells = mem_cells[:max_res]
    return mem_cells

@time_tracker
@memory_tracker
def execute(n_dimensions, num_patterns, clone_rate, mutate_rate, max_res, search_space):

    mem_cells = initialize_cells(max_res, n_dimensions, search_space)
    mem_cells = train_system(mem_cells, num_patterns, clone_rate, mutate_rate, max_res, search_space)
    best_cell = min(mem_cells, key=lambda x: x['score'])

    return best_cell

if __name__ == "__main__":
    
    algorithm_name = 'artificial immune recognition system'
    optimal_solution = 0
    # problem configuration
    problem_size = 3  # Change this to the number of variables in your problem
    num_patterns = 100
    search_space = [[-10, +10] for i in range(problem_size)]
    # algorithm configuration
    clone_rate = 10
    mutate_rate = 2.0
    max_res = 150
    # execute the algorithm
    for i in range(100):

        best = execute(problem_size, num_patterns, clone_rate, mutate_rate, max_res, search_space)
        solution = best['score']
        error = abs(optimal_solution - solution)
        arguments = best['vector']
        total_time = best['time']
        total_memory = best['memory']

        print("Done. Best Solution: c={}, v={}".format(solution, arguments))
        
        csv_file_name = 'DATA.csv'
        data = [algorithm_name, solution, error ,arguments, total_time, total_memory]

        write_to_csv(csv_file_name, data)

    calculate_average(csv_file_name,'method', algorithm_name, [ 'function_value','error','time_duration', 'total_memory'])
