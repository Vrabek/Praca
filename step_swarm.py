import random
import math
from Praca.problem_setup import objective_function, problem_configuration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np

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

def ACS_search(problem_size, search_space, max_iterations, num_ants=10, decay=0.1, c_heur=2.5, c_local_phero=0.1, c_greed=0.9):
    data = {'x1': [], 'x2': [], 'f': []}
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
        data['x1'].append(best['vector'][0])
        data['x2'].append(best['vector'][1])
        data['f'].append(best['cost'])

    return data

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

def BFOA_search(search_space, pop_size, elim_disp_steps= 1, repro_steps= 4, chem_steps= 70, swim_length= 4, step_size= 0.1, d_attr=0.1, w_attr=0.2, h_rep=0.2, w_rep=10, p_eliminate=0.25):
    data = {'x1': [], 'x2': [], 'f': []}
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
        data['x1'].append(best['vector'][0])
        data['x2'].append(best['vector'][1])
        data['f'].append(best['cost'])

    return data

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

def PSO_search(max_gens, search_space, vel_space, pop_size=50, max_vel=100, c1=2, c2=2):
    data = {'x1': [], 'x2': [], 'f': []}
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
        data['x1'].append(best['position'][0])
        data['x2'].append(best['position'][1])
        data['f'].append(best['cost'])

    return data

def ploting(first, second, third):
    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)

    # Tworzenie siatki 2D
    X1, X2 = np.meshgrid(x1, x2)

    # Definiowanie funkcji f = x1^2 + x2^2
    F = X1**2 + X2**2

    # Tworzenie wykresu 3D
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(X1, X2, F, cmap='viridis', alpha=0.5)
    ax.plot(first['x1'], first['x2'], first['f'], linewidth=2, color='blue')
    ax.scatter(0, 0, 0, linewidth=3 ,color='red')

    # Dodawanie tytułów i etykiet osi
    ax.set_title("System Mrowkowy", fontsize=16)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f")
    ax.view_init(90, -90)
    ax.set_zticks([])
    # Wyświetlanie wykresu

    ax1 = fig.add_subplot(132, projection='3d')  # 1x3 grid, second subplot
    ax1.plot_surface(X1, X2, F, cmap='viridis', alpha=0.5)
    ax1.plot(second['x1'], second['x2'], second['f'], linewidth=2, color='blue')
    ax1.scatter(0, 0, 0, linewidth=3 ,color='red')
     # Dodawanie tytułów i etykiet osi
    ax1.set_title("Algorytm Optymalizacji Poprzez Rozwoj Bakteryjny", fontsize=16)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("f")
    ax1.view_init(90, -90)
    ax1.set_zticks([])

    ax2 = fig.add_subplot(133, projection='3d')  # 1x3 grid, second subplot
    ax2.plot_surface(X1, X2, F, cmap='viridis', alpha=0.5)
    ax2.plot(third['x1'], third['x2'], third['f'], linewidth=2, color='blue')
    ax2.scatter(0, 0, 0, linewidth=3 ,color='red')
     # Dodawanie tytułów i etykiet osi
    ax2.set_title("Algorytm Optymalizacji Rojem Czastek", fontsize=16)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("f")
    ax2.view_init(90, -90)
    ax2.set_zticks([])

    plt.subplots_adjust(wspace=0)
    plt.show()

if __name__ == "__main__":

    # problem configuration
    problem_size, search_space, optimal_solution = problem_configuration()
    # algorithm configuration
    max_iterations = 100
    # execute the algorithm
    vel_space = [[-1, 1] for _ in range(problem_size)]

    data1 = ACS_search(problem_size, search_space, max_iterations)
    data2 = BFOA_search(search_space, max_iterations)
    data3 = PSO_search(max_iterations, search_space, vel_space)

    data1['x1'] = [5] + data1['x1']
    data1['x2'] = [5] + data1['x2']
    data1['f'] = [5] + data1['f']
    
    data2['x1'] = [5] + data2['x1']
    data2['x2'] = [5] + data2['x2']
    data2['f'] = [5] + data2['f']

    data3['x1'] = [5] + data3['x1']
    data3['x2'] = [5] + data3['x2']
    data3['f'] = [5] + data3['f']

    ploting(data1, data2, data3)
        