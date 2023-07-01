#zaimplementowane na podstawie pracy Clever Algorithms Nature-Inspired Programming Recipes
import random
import math

def euc_2d(c1, c2):
    return round(math.sqrt((c1[0] - c2[0])**2.0 + (c1[1] - c2[1])**2.0))

def cost(perm, cities):
    distance = 0
    for i, c1 in enumerate(perm):
        c2 = perm[0] if i == len(perm) - 1 else perm[i+1]
        distance += euc_2d(cities[c1], cities[c2])
    return distance

def stochastic_two_opt(permutation):
    perm = list(permutation)
    c1, c2 = random.randint(0, len(perm)-1), random.randint(0, len(perm)-1)
    exclude = [c1]
    exclude.append(perm[c1-1] if c1 > 0 else perm[-1])
    exclude.append(perm[c1+1] if c1 < len(perm)-1 else perm[0])
    while c2 in exclude:
        c2 = random.randint(0, len(perm)-1)
    if c2 < c1:
        c1, c2 = c2, c1
    perm[c1:c2] = reversed(perm[c1:c2])
    return perm

def local_search(best, cities, max_no_improv):
    count = 0
    while count < max_no_improv:
        candidate = {'vector': stochastic_two_opt(best['vector'])}
        candidate['cost'] = cost(candidate['vector'], cities)
        if candidate['cost'] < best['cost']:
            best = candidate
            count = 0
        else:
            count += 1
    return best

def construct_randomized_greedy_solution(cities, alpha):
    candidate = {}
    candidate['vector'] = [random.randint(0, len(cities)-1)]
    allCities = [i for i in range(len(cities))]
    while len(candidate['vector']) < len(cities):
        candidates = list(set(allCities) - set(candidate['vector']))
        costs = [euc_2d(cities[candidate['vector'][-1]], cities[i]) for i in candidates]
        c_min, c_max = min(costs), max(costs)
        rcl = [candidates[i] for i in range(len(candidates)) if costs[i] <= c_min + alpha*(c_max - c_min)]
        candidate['vector'].append(random.choice(rcl))
    candidate['cost'] = cost(candidate['vector'], cities)
    return candidate

def search(cities, max_iter, max_no_improv, alpha):
    best = None
    for i in range(max_iter):
        candidate = construct_randomized_greedy_solution(cities, alpha)
        candidate = local_search(candidate, cities, max_no_improv)
        if best is None or candidate['cost'] < best['cost']:
            best = candidate
        print(f"> iteration {i+1}, best={best['cost']}")
    return best

if __name__ == '__main__':
    # problem configuration
    berlin52 = [[565,575],[25,185],[345,750],[945,685],[845,655],
                [880,660],[25,230],[525,1000],[580,1175],[650,1130],[1605,620],
                [1220,580],[1465,200],[1530,5],[845,680],[725,370],[145,665],
                [415,635],[510,875],[560,365],[300,465],[520,585],[480,415],
                [835,625],[975,580],[1215,245],[1320,315],[1250,400],[660,180],
                [410,250],[420,555],[575,665],[1150,1160],[700,580],[685,595],
                [685,610],[770,610],[795,645],[720,635],[760,650],[475,960],
                [95,260],[875,920],[700,500],[555,815],[830,485],[1170,65],
                [830,610],[605,625],[595,360],[1340,725],[1740,245]]
    # algorithm configuration
    max_iter = 50
    max_no_improv = 50
    greediness_factor = 0.3
    # execute the algorithm
    best = search(berlin52, max_iter, max_no_improv, greediness_factor)
    print(f"Done. Best Solution: c={best['cost']}, v={best['vector']}")