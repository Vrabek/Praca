import random

from Praca.csv_utils import write_to_csv, calculate_average
from Praca.decorators import memory_tracker, time_tracker
from Praca.problem_setup import objective_function


def random_float(min_val, max_val):
    return min_val + random.random() * (max_val - min_val)

def generate_random_solution(search_space):
    return [random_float(min_val, max_val) for min_val, max_val in search_space]

def generate_initial_population(population_size, search_space):
    return [generate_random_solution(search_space) for _ in range(population_size)]

def evaluate_population(population):
    return [objective_function(candidate) for candidate in population]

def select_parents(population, num_parents):
    parents = []
    sorted_population = sorted(population, key=lambda x: objective_function(x))
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

def mutate(solution, mutation_rate, search_space):
    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            mutated_solution[i] = random_float(search_space[i][0], search_space[i][1])
    return mutated_solution

@time_tracker
@memory_tracker
def genetic_algorithm(search_space, population_size, num_generations, num_parents, num_offspring, mutation_rate):
    best = {}
    population = generate_initial_population(population_size, search_space)
    best_solution = None

    for generation in range(num_generations):
        fitness_values = evaluate_population(population)
        best_index = min(range(len(fitness_values)), key=fitness_values.__getitem__)
        best_solution = population[best_index]
        print(f"Iteration {generation+1}: Best solution = {best_solution}, Objective value = {fitness_values[best_index]}")

        parents = select_parents(population, num_parents)
        offspring = crossover(parents, num_offspring)
        mutated_offspring = [mutate(solution, mutation_rate, search_space) for solution in offspring]
        population = parents + mutated_offspring

    fitness_values = evaluate_population(population)
    best_index = min(range(len(fitness_values)), key=fitness_values.__getitem__)
    best_solution = population[best_index]

    best['cost'] = fitness_values[best_index]
    best['vector'] = best_solution

    return best


if __name__ == "__main__":

    algorithm_name = 'genetic algorithm'
    optimal_solution = 0
    # Problem configuration

    problem_size = 3
    search_space = [[-10, +10] for _ in range(problem_size)]

    # Algorithm configuration
    population_size = 100
    num_generations = 100
    num_parents = 50
    num_offspring = 50
    mutation_rate = 0.1

    # Execute the algorithm
   
    for i in range(100):

        best  = genetic_algorithm(search_space, population_size, num_generations, num_parents, num_offspring, mutation_rate)
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
