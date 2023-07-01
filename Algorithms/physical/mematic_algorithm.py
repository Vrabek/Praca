#zaimplementowane na podstawie pracy Clever Algorithms Nature-Inspired Programming Recipes
import random

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker
from Praca.problem_setup import objective_function, problem_configuration


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

@time_tracker
@memory_tracker
def search(max_gens, search_space, pop_size, p_cross, p_mut, max_local_gens, p_local, bits_per_param=16):

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

    return best

if __name__ == "__main__":
        
    algorithm_name = 'Algorytm Memetyczny'
    optimal_solution = 0
    # problem configuration
    problem_size, search_space, optimal_solution = problem_configuration()
    # algorithm configuration
    max_gens = 100
    pop_size = 100  
    p_cross = 0.98
    p_mut = 1.0/(problem_size*16)
    max_local_gens = 20
    p_local = 0.5
    # execute the algorithm
    for i in range(100):

        best = search(max_gens, search_space, pop_size, p_cross, p_mut, max_local_gens, p_local)
        solution = best['fitness']
        error = abs(optimal_solution - solution)
        arguments = best['vector']
        total_time = best['time']
        total_memory = best['memory']

        print("Done. Best Solution: c={}, v={}".format(solution, arguments))
        
        csv_file_name = 'DATA.csv'
        data = [algorithm_name, solution, error ,arguments, total_time, total_memory]

        write_to_csv(csv_file_name, data)

    calculate_average(csv_file_name,'method', algorithm_name, [ 'function_value','error','time_duration', 'total_memory'])
