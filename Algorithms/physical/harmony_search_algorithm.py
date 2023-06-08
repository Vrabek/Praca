import random

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker
from Praca.problem_setup import objective_function, problem_configuration


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

@time_tracker
@memory_tracker
def search(bounds, max_iter, mem_size, consid_rate, adjust_rate, range_value):

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

    return best

if __name__ == "__main__":
    
    algorithm_name = 'harmony search algorithm'
    
    # problem configuration
    problem_size, search_space, optimal_solution = problem_configuration()
    # algorithm configuration
    mem_size = 20
    consid_rate = 0.95
    adjust_rate = 0.7
    range_value = 0.05
    max_iter = 100
    # execute the algorithm
    for i in range(100):

        best = search(search_space, max_iter, mem_size, consid_rate, adjust_rate, range_value)
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
