import random
import math
from Praca.problem_setup import objective_function, problem_configuration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np

def random_vector_1(n_dimensions, search_space):
    return [random.uniform(search_space[i][0], search_space[i][1]) for i in range(n_dimensions)]

def create_cell(vector):
    return {'vector': vector, 'score': objective_function(vector)}

def initialize_cells(n_cells, n_dimensions, search_space):
    return [create_cell(random_vector_1(n_dimensions, search_space)) for _ in range(n_cells)]

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
        cell['resources'] = clone_rate / cell['score']
    pool.sort(key=lambda x: x['resources'], reverse=True)
    total_resources = sum(cell['resources'] for cell in pool)
    while total_resources > max_res:
        cell = pool.pop()
        total_resources -= cell['resources']

def train_system(mem_cells, num_patterns, clone_rate, mutate_rate, max_res, search_space):
    data = {'x1': [], 'x2': [], 'f': []}
    for i in range(num_patterns):
        best_match = min(mem_cells, key=lambda x: x['score'])
        print(f"Iteration {i+1}, Best score: {best_match['score']}, Best vector: {best_match['vector']}")
        data['x1'].append(best_match['vector'][0])
        data['x2'].append(best_match['vector'][1])
        data['f'].append(best_match['score'])
        pool = create_arb_pool(best_match, clone_rate, mutate_rate, search_space)
        competition_for_resources(pool, clone_rate, max_res)
        mem_cells.extend(pool)
        mem_cells.sort(key=lambda x: x['score'])
        if len(mem_cells) > max_res:
            mem_cells = mem_cells[:max_res]
    return mem_cells, data

def AIRS_search(n_dimensions, num_patterns, clone_rate, mutate_rate, max_res, search_space):

    mem_cells = initialize_cells(max_res, n_dimensions, search_space)
    mem_cells, data = train_system(mem_cells, num_patterns, clone_rate, mutate_rate, max_res, search_space)

    return data

def decode(bitstring, search_space, bits_per_param):
    vector = []
    for i, bounds in enumerate(search_space):
        off, s = i*bits_per_param, 0.0
        param = bitstring[off:(off+bits_per_param)][::-1]
        for j in range(len(param)):
            s += (1.0 if param[j]=='1' else 0.0) * (2.0 ** float(j))
        min_b, max_b = bounds
        vector.append(min_b + ((max_b-min_b)/((2.0**bits_per_param)-1.0)) * s)
    return vector

def evaluate(pop, search_space, bits_per_param):
    for p in pop:
        p['vector'] = decode(p['bitstring'], search_space, bits_per_param)
        p['cost'] = objective_function(p['vector'])

def random_bitstring(num_bits):
    return ''.join('1' if random.random() < 0.5 else '0' for _ in range(num_bits))

def point_mutation(bitstring, rate):
    child = ""
    for i in range(len(bitstring)):
        bit = bitstring[i]
        child += '0' if (bit=='1') else '1' if random.random()<rate else bit
    return child

def calculate_mutation_rate(antibody, mutate_factor=-2.5):
    return math.exp(mutate_factor * antibody['affinity'])

def num_clones(pop_size, clone_factor):
    return int(pop_size * clone_factor)

def calculate_affinity(pop):
    pop.sort(key=lambda x: x['cost'])
    range_c = pop[-1]['cost'] - pop[0]['cost']
    if range_c == 0.0:
        for p in pop:
            p['affinity'] = 1.0
    else:
        for p in pop:
            p['affinity'] = 1.0-(p['cost']/range_c)

def clone_and_hypermutate(pop, clone_factor):
    clones = []
    num_clones_c = num_clones(len(pop), clone_factor)
    calculate_affinity(pop)
    for antibody in pop:
        m_rate = calculate_mutation_rate(antibody)
        for _ in range(num_clones_c):
            clone = {}
            clone['bitstring'] = point_mutation(antibody['bitstring'], m_rate)
            clones.append(clone)
    return clones  

def random_insertion(search_space, pop, num_rand, bits_per_param):
    if num_rand == 0:
        return pop
    rands = [{'bitstring': random_bitstring(len(search_space)*bits_per_param)} for _ in range(num_rand)]
    evaluate(rands, search_space, bits_per_param)
    return sorted(pop+rands, key=lambda x: x['cost'])[:len(pop)]

def CSA_search(search_space, max_gens, pop_size, clone_factor=0.1, num_rand=2, bits_per_param=16):
    data = {'x1': [], 'x2': [], 'f': []}
    pop = [{'bitstring': random_bitstring(len(search_space)*bits_per_param)} for _ in range(pop_size)]
    evaluate(pop, search_space, bits_per_param)
    best = min(pop, key=lambda x: x['cost'])
    for gen in range(max_gens):
        clones = clone_and_hypermutate(pop, clone_factor)
        evaluate(clones, search_space, bits_per_param)
        pop = sorted(pop+clones, key=lambda x: x['cost'])[:pop_size]
        pop = random_insertion(search_space, pop, num_rand, bits_per_param)
        best = min(pop + [best], key=lambda x: x['cost'])
        print(f" > gen {gen+1}, f={best['cost']}, s={best['vector']}")
        data['x1'].append(best['vector'][0])
        data['x2'].append(best['vector'][1])
        data['f'].append(best['cost'])

    return data

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


def DCA_search(search_space, max_iter, swarm_size=20, c1=2, c2=2):
    best = {}
    data = {'x1': [], 'x2': [], 'f': []}
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
        data['x1'].append(best['vector'][0])
        data['x2'].append(best['vector'][1])
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
    fig.suptitle('Algorytmy Immunologiczne dla 1000 iteracji', fontsize=16)
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(X1, X2, F, cmap='viridis', alpha=0.5)
    ax.plot(first['x1'], first['x2'], first['f'], linewidth=2, color='blue')
    ax.scatter(0, 0, 0, linewidth=3 ,color='red')

    # Dodawanie tytułów i etykiet osi
    ax.set_title("System Sztucznego Rozpoznawania\n Ukladu Immunologicznego", fontsize=16)
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
    ax1.set_title("Algorytm Selekcji Klonalnej", fontsize=16)
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
    ax2.set_title("Algorytm Komorek Dendrytycznych", fontsize=16)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("f")
    ax2.view_init(90, -90)
    ax2.set_zticks([])

    plt.subplots_adjust(wspace=0)
    plt.show()

if __name__ == '__main__':
    problem_size, search_space, optimal_solution = problem_configuration()
    # algorithm configuration
    max_iter = 1000
    clone_rate = 10
    mutate_rate = 2.0
    max_res = 150
    # execute the algorithm

    data1 = AIRS_search(problem_size, max_iter, clone_rate, mutate_rate, max_res, search_space)
    data2 = CSA_search(search_space, 200, 200)
    data3 = DCA_search(search_space, max_iter)

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
 