import math
import random


from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker

def objective_function(vector):
    return sum(x ** 2.0 for x in vector)

def random_vector(minmax):
    return [minmax[i][0] + ((minmax[i][1] - minmax[i][0]) * random.random()) for i in range(len(minmax))]

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

@time_tracker
@memory_tracker
def search(max_gens, search_space, pop_size, num_children):
    
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
    
    return best

if __name__ == "__main__":

    algorithm_name = 'evolution strategies algorithm search'
    optimal_solution = 0
    # problem configuration
    problem_size = 3
    search_space = [[-10, +10] for _ in range(problem_size)]
    # algorithm configuration
    max_gens = 100
    pop_size = 30
    num_children = 20
    # execute the algorithm
    for i in range(100):

        best  = search(max_gens, search_space, pop_size, num_children)
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
