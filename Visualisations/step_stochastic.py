
from Praca.problem_setup import objective_function, problem_configuration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np


def random_vector(minmax):
    return [minmax[i][0] + ((minmax[i][1] - minmax[i][0]) * random.random()) for i in range(len(minmax))]


def random_search(search_space, max_iter):
    best = None
    data = {'x1': [], 'x2': [], 'f': []}
    for iter in range(max_iter):
        candidate = {}
        candidate['vector'] = random_vector(search_space)
        candidate['cost'] = objective_function(candidate['vector'])
        if best is None or candidate['cost'] < best['cost']:
            best = candidate
        print(" > iteration={}, argument={} best={}".format(iter+1,best['vector'], best['cost']))
        data['x1'].append(best['vector'][0])
        data['x2'].append(best['vector'][1])
        data['f'].append(best['cost'])

    return data


def local_search(best, tabu_list, search_space, max_no_improv):
    count = 0
    while count < max_no_improv:
        candidate = random_vector(search_space)
        if candidate not in tabu_list and (best is None or objective_function(candidate) < objective_function(best)):
            best = candidate
            count = 0
        else:
            count += 1
        tabu_list.append(candidate)
        if len(tabu_list) > max_tabu_size:
            tabu_list.pop(0)
    return best


def tabu_search(search_space, max_iter, max_no_improv, max_tabu_size):
    data = {'x1': [], 'x2': [], 'f': []}
    best_cost = None
    tabu_list = []
    for _ in range(max_iter):
        candidate = random_vector(search_space)
        candidate = local_search(candidate, tabu_list, search_space, max_no_improv)
        if best_cost is None or objective_function(candidate) < objective_function(best_cost):
            best_cost = candidate
        
        print(f"Iteration: Arguments = {best_cost}, Objective Function Value = {objective_function(best_cost)}")
        data['x1'].append(best_cost[0])
        data['x2'].append(best_cost[1])
        data['f'].append(objective_function(best_cost))

    return data


def construct_greedy_solution(search_space, alpha):
    candidate = {}
    candidate['vector'] = random_vector(search_space)
    candidate['cost'] = objective_function(candidate['vector'])
    return candidate

def local_search_grasp(best, max_no_improv):
    count = 0
    while count < max_no_improv:
        candidate = construct_greedy_solution(search_space, alpha=0)
        candidate['cost'] = objective_function(candidate['vector'])
        if candidate['cost'] < best['cost']:
            best = candidate
            count = 0
        else:
            count += 1
    return best

def GRASP_search(search_space, problem_size, max_iter, max_no_improv):
    best = None
    data = {'x1': [], 'x2': [], 'f': []}
    for iter in range(max_iter):
        candidate = construct_greedy_solution(search_space, alpha=0.3)
        candidate = local_search_grasp(candidate, max_no_improv)
        if best is None or candidate['cost'] < best['cost']:
            best = candidate
        print(f"Iteration {iter + 1}: Arguments = {best['vector']}, Objective Function Value = {best['cost']}")
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
    ax = fig.add_subplot(131, projection='3d')
    fig.suptitle('Algorytmy Stochastyczne dla 1000 iteracji', fontsize=16)
    ax.plot_surface(X1, X2, F, cmap='viridis', alpha=0.5)
    ax.plot(first['x1'], first['x2'], first['f'], linewidth=2, color='blue')
    ax.scatter(0, 0, 0, linewidth=3 ,color='red')

    # Dodawanie tytułów i etykiet osi
    ax.set_title("Losowe Poszukiwanie", fontsize=16)
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
    ax1.set_title("Przeszukiwanie z Zakazem", fontsize=16)
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
    ax2.set_title("Zachlanny Algorytm Losowego \nAdaptacyjnego Przeszukiwania", fontsize=16)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("f")
    ax2.view_init(90, -90)
    ax2.set_zticks([])

    plt.subplots_adjust(wspace=0)
    plt.show()

if __name__ == "__main__":

    algorithm_name = 'random search'
    
    # problem configuration
    problem_size, search_space, optimal_solution = problem_configuration()
    # algorithm configuration
    max_iter = 1000
    max_no_improv = 50
    max_tabu_size = 10
    # execute the algorithm
    data1 = random_search([[-7,7],[-7,7]], max_iter)
    data2 = tabu_search(search_space, max_iter, max_no_improv, max_tabu_size)
    data3 = GRASP_search(search_space, problem_size, max_iter, max_no_improv)



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


