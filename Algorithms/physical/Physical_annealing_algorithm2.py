#zaimplementowane na podstawie pracy Clever Algorithms Nature-Inspired Programming Recipes
import math
import random

def euc_2d(c1, c2):
    return round(math.sqrt((c1[0] - c2[0])**2.0 + (c1[1] - c2[1])**2.0))

def cost(permutation, cities):
    distance = 0
    for i, c1 in enumerate(permutation):
        c2 = permutation[0] if i == len(permutation)-1 else permutation[i+1]
        distance += euc_2d(cities[c1], cities[c2])
    return distance

def random_permutation(cities):
    perm = list(range(len(cities)))
    for i in range(len(perm)):
        r = random.randint(0, len(perm)-i-1) + i
        perm[r], perm[i] = perm[i], perm[r]
    return perm

def stochastic_two_opt(perm):
    c1, c2 = random.randint(0, len(perm)-1), random.randint(0, len(perm)-1)
    exclude = [c1]
    exclude.append(len(perm)-1 if c1==0 else c1-1)
    exclude.append(0 if c1==len(perm)-1 else c1+1)
    while c2 in exclude:
        c2 = random.randint(0, len(perm)-1)
    if c2 < c1:
        c1, c2 = c2, c1
    perm[c1:c2] = reversed(perm[c1:c2])
    return perm

def create_neighbor(current, cities):
    candidate = {}
    candidate['vector'] = list(current['vector'])
    candidate['vector'] = stochastic_two_opt(candidate['vector'])
    candidate['cost'] = cost(candidate['vector'], cities)
    return candidate

def should_accept(candidate, current, temp):
    if candidate['cost'] <= current['cost']:
        return True
    return math.exp((current['cost'] - candidate['cost']) / temp) > random.random()

def search(cities, max_iter, max_temp, temp_change):
    current = {'vector': random_permutation(cities)}
    current['cost'] = cost(current['vector'], cities)
    temp, best = max_temp, current
    for iter in range(max_iter):
        candidate = create_neighbor(current, cities)
        temp *= temp_change
        if should_accept(candidate, current, temp):
            current = candidate
        if candidate['cost'] < best['cost']:
            best = candidate
        if (iter+1) % 10 == 0:
            print(f" > iteration {iter+1}, temp={temp}, best={best['cost']}")
    return best

if __name__ == "__main__":
    berlin52 = [[565,575],[25,185],[345,750],[945,685],[845,655],
    [880,660],[25,230],[525,1000],[580,1175],[650,1130],[1605,620],
    [1220,580],[1465,200],[1530,5],[845,680],[725,370],[145,665],
    [415,635],[510,875],[560,365],[300,465],[520,585],[480,415],
    [835,625],[975,580],[1215,245],[1320,315],[1250,400],[660,180],
    [410,250],[420,555],[575,665],[1150,1160],[700,580],[685,595],
    [685,610],[770,610],[795,645],[720,635],[760,650],[475,960],
    [95,260],[875,920],[700,500],[555,815],[830,485],[1170,65],
    [830,610],[605,625],[595,360],[1340,725],[1740,245]]
    max_iterations = 2000
    max_temp = 100000.0
    temp_change = 0.98
    best = search(berlin52, max_iterations, max_temp, temp_change)
    print(f"Done. Best Solution: c={best['cost']}, v={best['vector']}")