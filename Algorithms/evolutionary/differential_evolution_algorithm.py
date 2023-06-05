import random

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker

def objective_function(vector):
    return sum(x ** 2.0 for x in vector)

def random_vector(minmax):
    return [minmax[i][0] + ((minmax[i][1] - minmax[i][0]) * random.random()) for i in range(len(minmax))]

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

@time_tracker
@memory_tracker
def search(max_gens, search_space, pop_size, f, cr):
    
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

    return best

if __name__ == '__main__':

    algorithm_name = 'diffrential evolution algorithm'
    optimal_solution = 0
    # problem configuration
    problem_size = 3
    search_space = [[-10, +10] for i in range(problem_size)]
    # algorithm configuration
    max_gens = 100
    pop_size = 10 * problem_size
    weightf = 0.8
    crossf = 0.9
    # execute the algorithm
    for i in range(100):

        best = search(max_gens, search_space, pop_size, weightf, crossf)
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
