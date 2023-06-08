import random
import math

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker
from Praca.problem_setup import objective_function, problem_configuration


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

@time_tracker
@memory_tracker
def search(search_space, max_gens, pop_size, clone_factor, num_rand, bits_per_param=16):

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

    return best

if __name__ == "__main__":
    
    algorithm_name = 'colonal selection algorithm'
    # problem configuration
    problem_size, search_space, optimal_solution = problem_configuration()
    # algorithm configuration
    max_gens = 100
    pop_size = 100
    clone_factor = 0.1
    num_rand = 2
    # execute the algorithm

    for i in range(100):

        best = search(search_space, max_gens, pop_size, clone_factor, num_rand)
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
