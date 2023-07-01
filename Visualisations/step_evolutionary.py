import random
import math
from Praca.problem_setup import objective_function, problem_configuration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np

def random_vector(minmax):
    return [5+random.uniform(-1,1), 5+random.uniform(-1,1)]

def de_rand_1_bin(p0, p1, p2, p3, f, cr, search_space):
    sample = {'vector': [0.0] * len(p0['vector'])}
    cut = random.randint(1, len(sample['vector']) - 1)
    for i in range(len(sample['vector'])):
        sample['vector'][i] = p0['vector'][i]
        if i == cut or random.random() < cr:
            v = p3['vector'][i] + f * (p1['vector'][i] - p2['vector'][i])
            v = search_space[i][0] if v < search_space[i][0] else v
            v = search_space[i][1] if v > search_space[i][1] else v
            sample['vector'][i] = v
    return sample

def select_parents(pop, current):
    p1, p2, p3 = random.randint(0, len(pop) - 1), random.randint(0, len(pop) - 1), random.randint(0, len(pop) - 1)
    while p1 == current:
        p1 = random.randint(0, len(pop) - 1)
    while p2 == current or p2 == p1:
        p2 = random.randint(0, len(pop) - 1)
    while p3 == current or p3 == p1 or p3 == p2:
        p3 = random.randint(0, len(pop) - 1)
    return [p1, p2, p3]

def create_children(pop, minmax, f, cr):
    children = []
    for i, p0 in enumerate(pop):
        p1, p2, p3 = select_parents(pop, i)
        children.append(de_rand_1_bin(p0, pop[p1], pop[p2], pop[p3], f, cr, minmax))
    return children

def select_population(parents, children):
    return [children[i] if children[i]['cost'] <= parents[i]['cost'] else parents[i] for i in range(len(parents))]


def DEA_search(max_gens, search_space, pop_size=20, f=0.8, cr=0.9):
    data = {'x1': [], 'x2': [], 'f': []}
    pop = [{'vector': random_vector(search_space)} for _ in range(pop_size)]
    for c in pop:
        c['cost'] = objective_function(c['vector'])
    best = sorted(pop, key=lambda x: x['cost'])[0]
    for gen in range(max_gens):
        children = create_children(pop, search_space, f, cr)
        for c in children:
            c['cost'] = objective_function(c['vector'])
        pop = select_population(pop, children)
        pop = sorted(pop, key=lambda x: x['cost'])
        best = pop[0] if pop[0]['cost'] < best['cost'] else best
        print(f"> gen {gen+1}, fitness={best['cost']}")

        
        data['x1'].append(best['vector'][0])
        data['x2'].append(best['vector'][1])
        data['f'].append(best['cost'])

    return data

def random_gaussian(mean=0.0, stdev=1.0):
    u1 = u2 = w = 0
    while w >= 1 or w == 0:
        u1 = 2 * random.random() - 1
        u2 = 2 * random.random() - 1
        w = u1 * u1 + u2 * u2
    w = math.sqrt((-2.0 * math.log(w)) / w)
    return mean + (u2 * w) * stdev

def mutate_problem(vector, stdevs, search_space):
    child = [0]*len(vector)
    for i, v in enumerate(vector):
        child[i] = v + stdevs[i] * random_gaussian()
        child[i] = max(min(child[i], search_space[i][1]), search_space[i][0])
    return child

def mutate_strategy(stdevs):
    tau = math.sqrt(2.0*len(stdevs))**-1.0
    tau_p = math.sqrt(2.0*math.sqrt(len(stdevs)))**-1.0
    child = [stdevs[i] * math.exp(tau_p*random_gaussian() + tau*random_gaussian()) for i in range(len(stdevs))]
    return child

def mutate(par, minmax):
    child = {}
    child['vector'] = mutate_problem(par['vector'], par['strategy'], minmax)
    child['strategy'] = mutate_strategy(par['strategy'])
    return child

def init_population(minmax, pop_size):
    strategy = [[0, (minmax[i][1]-minmax[i][0]) * 0.05] for i in range(len(minmax))]
    pop = [{'vector': random_vector(minmax), 'strategy': random_vector(strategy)} for _ in range(pop_size)]
    for c in pop:
        c['fitness'] = objective_function(c['vector'])
    return pop


def ESA_search(max_gens, search_space, pop_size=30, num_children=20):
    data = {'x1': [], 'x2': [], 'f': []}
    population = init_population(search_space, pop_size)
    population.sort(key=lambda x: x['fitness'])
    best = population[0]
    for gen in range(max_gens):
        children = [mutate(population[i], search_space) for i in range(num_children)]
        for c in children:
            c['fitness'] = objective_function(c['vector'])
        union = children + population
        union.sort(key=lambda x: x['fitness'])
        if union[0]['fitness'] < best['fitness']:
            best = union[0]
        population = union[:pop_size]
        print(f" > gen {gen}, fitness={best['fitness']}")

        data['x1'].append(best['vector'][0])
        data['x2'].append(best['vector'][1])
        data['f'].append(best['fitness'])
    
    return data

def random_float(min_val, max_val):
    return min_val + random.random() * (max_val - min_val)

def generate_random_solution(search_space):
    return [5,5]

def generate_initial_population(population_size, search_space):
    return [generate_random_solution(search_space) for _ in range(population_size)]

def evaluate_population(population):
    return [objective_function(candidate) for candidate in population]

def select_parents_gen(population, num_parents):
    parents = []
    sorted_population = sorted(population, key=lambda x: objective_function(x))
    num_parents = min(num_parents, len(sorted_population))  # Nowy wiersz
    for i in range(num_parents):
        parents.append(sorted_population[i])
    return parents

def crossover(parents, num_offspring):
    offspring = []
    num_parents = len(parents)
    for _ in range(num_offspring):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = [0] * len(parent1)
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        offspring.append(child)
    return offspring

def mutate_gen(solution, mutation_rate, search_space):
    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            mutated_solution[i] = random_float(search_space[i][0], search_space[i][1])
    return mutated_solution

def genetic_algorithm(search_space, population_size=100, num_generations=100, num_parents=50, num_offspring=50, mutation_rate=0.1):
    best = {}
    data = {'x1': [], 'x2': [], 'f': []}
    population = generate_initial_population(population_size, search_space)
    best_solution = None

    for generation in range(num_generations):
        fitness_values = evaluate_population(population)
        best_index = min(range(len(fitness_values)), key=fitness_values.__getitem__)
        best_solution = population[best_index]
        print(f"Iteration {generation+1}: Best solution = {best_solution}, Objective value = {fitness_values[best_index]}")
        data['x1'].append(best_solution[0])
        data['x2'].append(best_solution[1])
        data['f'].append(fitness_values[best_index])
        parents = select_parents_gen(population, num_parents)
        offspring = crossover(parents, num_offspring)
        mutated_offspring = [mutate_gen(solution, mutation_rate, search_space) for solution in offspring]
        population = parents + mutated_offspring

    fitness_values = evaluate_population(population)
    best_index = min(range(len(fitness_values)), key=fitness_values.__getitem__)
    best_solution = population[best_index]

    best['cost'] = fitness_values[best_index]
    best['vector'] = best_solution

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
    fig.suptitle('Algorytmy Ewolucyjne dla 1000 iteracji', fontsize=16)
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(X1, X2, F, cmap='viridis', alpha=0.5)
    ax.plot(first['x1'], first['x2'], first['f'], linewidth=2, color='blue')
    ax.scatter(0, 0, 0, linewidth=3 ,color='red')

    # Dodawanie tytułów i etykiet osi
    ax.set_title("Algorytm Ewolucji Roznicowej", fontsize=16)
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
    ax1.set_title("Algorytm Strategii Ewolucyjnych", fontsize=16)
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
    ax2.set_title("Algorytm Genetyczny", fontsize=16)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("f")
    ax2.view_init(90, -90)
    ax2.set_zticks([])
    
    plt.subplots_adjust(wspace=0)
    #plt.savefig('demo.png', dpi=300, transparent=False, bbox_inches = 'tight', pad_inches = 0.1)

    plt.show()
    
if __name__ == '__main__':

    # problem configuration
    problem_size, search_space, optimal_solution = problem_configuration()
    # algorithm configuration
    max_gens = 1000
    pop_size = 10 * problem_size
    weightf = 0.8
    crossf = 0.9
    # execute the algorithm

    pop_size = 30
    num_children = 20

    data1 = DEA_search(max_gens, search_space)
    data2 = ESA_search(max_gens, search_space)
    data3 = genetic_algorithm(search_space, max_gens, max_gens)

    data1['x1'] = [5] + data1['x1']
    data1['x2'] = [5] + data1['x2']
    data1['f'] = [5] + data1['f']
    
    data2['x1'] = [5] + data2['x1']
    data2['x2'] = [5] + data2['x2']
    data2['f'] = [5] + data2['f']

    ploting(data1, data2, data3)
