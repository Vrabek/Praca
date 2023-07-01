import random
import math
from Praca.problem_setup import objective_function, problem_configuration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np

def rand_in_bounds(min, max):
    return min + ((max-min) * random.random())

def random_vector(search_space):
    return [rand_in_bounds(i[0], i[1]) for i in search_space]

def create_random_harmony(search_space):
    harmony = {}
    harmony["vector"] = random_vector(search_space)
    harmony["fitness"] = objective_function(harmony["vector"])
    return harmony

def initialize_harmony_memory(search_space, mem_size, factor=3):
    memory = [create_random_harmony(search_space) for _ in range(mem_size * factor)]
    memory.sort(key=lambda x: x["fitness"])
    return memory[:mem_size]

def create_harmony(search_space, memory, consid_rate, adjust_rate, range_value):
    vector = [0]*len(search_space)
    for i in range(len(search_space)):
        if random.random() < consid_rate:
            value = memory[random.randint(0, len(memory)-1)]["vector"][i]
            if random.random() < adjust_rate:
                value += range_value*rand_in_bounds(-1.0, 1.0)
            value = max(min(value, search_space[i][1]), search_space[i][0])
            vector[i] = value
        else:
            vector[i] = rand_in_bounds(search_space[i][0], search_space[i][1])
    return {"vector": vector}


def HS_search(bounds, max_iter, mem_size, consid_rate, adjust_rate, range_value):
    data = {'x1': [], 'x2': [], 'f': []}
    memory = initialize_harmony_memory(bounds, mem_size)
    best = memory[0]
    for iter in range(max_iter):
        harm = create_harmony(bounds, memory, consid_rate, adjust_rate, range_value)
        harm["fitness"] = objective_function(harm["vector"])
        if harm["fitness"] < best["fitness"]:
            best = harm
        memory.append(harm)
        memory.sort(key=lambda x: x["fitness"])
        memory.pop()
        print(f" > iteration={iter}, fitness={best['fitness']}")
        data['x1'].append(best['vector'][0])
        data['x2'].append(best['vector'][1])
        data['f'].append(best['fitness'])

    return data



def random_bitstring(num_bits):
    return ''.join('1' if random.random()<0.5 else '0' for _ in range(num_bits))

def decode(bitstring, search_space, bits_per_param):
    vector = []
    for i, bounds in enumerate(search_space):
        off = i * bits_per_param
        param = bitstring[off:off+bits_per_param][::-1]
        sum_ = sum((1.0 if bit=='1' else 0.0) * (2.0 ** j) for j, bit in enumerate(param))
        min_, max_ = bounds
        vector.append(min_ + ((max_-min_)/((2.0**bits_per_param)-1.0)) * sum_)
    return vector

def fitness(candidate, search_space, param_bits):
    candidate['vector'] = decode(candidate['bitstring'], search_space, param_bits)
    candidate['fitness'] = objective_function(candidate['vector'])

def binary_tournament(pop):
    i, j = random.sample(range(len(pop)), 2)
    return pop[i] if pop[i]['fitness'] < pop[j]['fitness'] else pop[j]

def point_mutation(bitstring, rate=None):
    if rate is None:
        rate = 1.0 / len(bitstring)
    return ''.join((bit if random.random()>=rate else ('1' if bit=='0' else '0')) for bit in bitstring)


def crossover(parent1, parent2, rate):
    if random.random()>=rate:
        return parent1
    return ''.join(parent1[i] if random.random()<0.5 else parent2[i] for i in range(len(parent1)))

def reproduce(selected, pop_size, p_cross, p_mut):
    children = []  
    for i, p1 in enumerate(selected):
        p2 = selected[i+1] if i % 2 == 0 else selected[i-1]
        if i == len(selected)-1:
            p2 = selected[0]
        child = {}
        child['bitstring'] = crossover(p1['bitstring'], p2['bitstring'], p_cross)
        child['bitstring'] = point_mutation(child['bitstring'], p_mut)
        if len(children) >= pop_size:
            break
        children.append(child)
    return children

def bitclimber(child, search_space, p_mut, max_local_gens, bits_per_param):
    current = child
    for _ in range(max_local_gens):
        candidate = {}
        candidate['bitstring'] = point_mutation(current['bitstring'], p_mut)
        fitness(candidate, search_space, bits_per_param)
        if candidate['fitness'] <= current['fitness']:
            current = candidate
    return current

def MA_search(max_gens, search_space, pop_size=100, p_cross=0.98, p_mut=1/36, max_local_gens=100, p_local=0.5, bits_per_param=16):
    data = {'x1': [], 'x2': [], 'f': []}
    pop = [{'bitstring': random_bitstring(len(search_space)*bits_per_param)} for _ in range(pop_size)]
    for candidate in pop:
        fitness(candidate, search_space, bits_per_param)
    pop.sort(key=lambda x: x['fitness'])
    best = pop[0]

    for gen in range(max_gens):
        selected = [binary_tournament(pop) for _ in range(pop_size)]
        children = reproduce(selected, pop_size, p_cross, p_mut)
        for cand in children:
            fitness(cand, search_space, bits_per_param)
        pop = []
        for child in children:
            if random.random() < p_local:
                child = bitclimber(child, search_space, p_mut, max_local_gens, bits_per_param)
            pop.append(child)
        pop.sort(key=lambda x: x['fitness'])
        if pop[0]['fitness'] <= best['fitness']:
            best = pop[0]
        print(f">gen={gen}, f={best['fitness']}, b={best['bitstring']}")
        data['x1'].append(best['vector'][0])
        data['x2'].append(best['vector'][1])
        data['f'].append(best['fitness'])

    return data

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

def SA_search(search_space, max_iter, max_temp=10000, temp_change=0.98):
    data = {'x1': [], 'x2': [], 'f': []}
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
    fig.suptitle('Algorytmy Fizyczne dla 1000 iteracji', fontsize=16)
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(X1, X2, F, cmap='viridis', alpha=0.5)
    ax.plot(first['x1'], first['x2'], first['f'], linewidth=2, color='blue')
    ax.scatter(0, 0, 0, linewidth=3 ,color='red')

    # Dodawanie tytułów i etykiet osi
    ax.set_title("Algorytm Harmonijnego Przeszukiwania", fontsize=16)
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
    ax1.set_title("Algorytm Memetyczny", fontsize=16)
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
    ax2.set_title("Algorytm Symulowanego Wyzarzania", fontsize=16)
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
    mem_size = 20
    consid_rate = 0.95
    adjust_rate = 0.7
    range_value = 0.05
    max_iter = 1000
    # execute the algorithm


    data1 = HS_search(search_space, max_iter, mem_size, consid_rate, adjust_rate, range_value)
    data2 = MA_search(max_iter, search_space)
    data3 = SA_search(search_space, max_iter)

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
