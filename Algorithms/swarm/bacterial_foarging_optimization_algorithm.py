import random
import math

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker

def objective_function(vector):
    return sum(x ** 2.0 for x in vector)

def random_vector(minmax):
    return [minmax[i][0] + ((minmax[i][1] - minmax[i][0]) * random.random()) for i in range(len(minmax))]

def generate_random_direction(problem_size):
    bounds = [[-1.0,1.0] for _ in range(problem_size)]
    return random_vector(bounds)

def compute_cell_interaction(cell, cells, d, w):
    sum_val = 0.0
    for other in cells:
        diff = sum((cell['vector'][i] - other['vector'][i])**2.0 for i in range(len(cell['vector'])))
        sum_val += d * math.exp(w * diff)
    return sum_val

def attract_repel(cell, cells, d_attr, w_attr, h_rep, w_rep):
    attract = compute_cell_interaction(cell, cells, -d_attr, -w_attr)
    repel = compute_cell_interaction(cell, cells, h_rep, -w_rep)
    return attract + repel

def evaluate(cell, cells, d_attr, w_attr, h_rep, w_rep):
    cell['cost'] = objective_function(cell['vector'])
    cell['inter'] = attract_repel(cell, cells, d_attr, w_attr, h_rep, w_rep)
    cell['fitness'] = cell['cost'] + cell['inter']

def tumble_cell(search_space, cell, step_size):
    step = generate_random_direction(len(search_space))  
    vector = [cell['vector'][i] + step_size * step[i] for i in range(len(search_space))]
    for i in range(len(vector)):
        vector[i] = max(min(vector[i], search_space[i][1]), search_space[i][0])
    return {'vector': vector}

def chemotaxis(cells, search_space, chem_steps, swim_length, step_size, d_attr, w_attr, h_rep, w_rep): 
    best = None
    for _ in range(chem_steps):
        moved_cells = []
        for i, cell in enumerate(cells):
            sum_nutrients = 0.0
            evaluate(cell, cells, d_attr, w_attr, h_rep, w_rep)
            if best is None or cell['cost'] < best['cost']:
                best = cell
            sum_nutrients += cell['fitness']
            for _ in range(swim_length):
                new_cell = tumble_cell(search_space, cell, step_size)
                evaluate(new_cell, cells, d_attr, w_attr, h_rep, w_rep)
                if new_cell['fitness'] > cell['fitness']:
                    break
                cell = new_cell
                sum_nutrients += cell['fitness']
            cell['sum_nutrients'] = sum_nutrients
            moved_cells.append(cell)
        cells = moved_cells
    return best, cells

@time_tracker
@memory_tracker
def search(search_space, pop_size, elim_disp_steps, repro_steps, chem_steps, swim_length, step_size, d_attr, w_attr, h_rep, w_rep, p_eliminate):

    cells = [{'vector': random_vector(search_space)} for _ in range(pop_size)]
    best = None
    for _ in range(elim_disp_steps):
        for _ in range(repro_steps):
            c_best, cells = chemotaxis(cells, search_space, chem_steps, swim_length, step_size, d_attr, w_attr, h_rep, w_rep) 
            if best is None or c_best['cost'] < best['cost']:
                best = c_best
            cells.sort(key=lambda x: x['sum_nutrients'])
            cells = cells[:pop_size//2] * 2
        for cell in cells:
            if random.random() <= p_eliminate:
                cell['vector'] = random_vector(search_space)

    return best


if __name__ == "__main__":
    
    algorithm_name = 'bacterial foarging optimization algorithm'
    optimal_solution = 0
    # problem configuration
    problem_size = 3
    search_space = [[-10, 10] ] * problem_size
    # algorithm configuration
    pop_size = 100
    step_size = 0.1 # Ci
    elim_disp_steps = 1 # Ned
    repro_steps = 4 # Nre
    chem_steps = 70 # Nc
    swim_length = 4 # Ns
    p_eliminate = 0.25 # Ped
    d_attr = 0.1
    w_attr = 0.2 
    h_rep = d_attr
    w_rep = 10
    # execute the algorithm
    for i in range(100):

        best = search(search_space, pop_size, elim_disp_steps, repro_steps, chem_steps, swim_length, step_size, d_attr, w_attr, h_rep, w_rep, p_eliminate)
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
