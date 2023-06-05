import math
import random

def euc_2d(c1, c2):
    return round(math.sqrt((c1[0] - c2[0])**2.0 + (c1[1] - c2[1])**2.0))

def cost(perm, cities):
    distance = 0
    for i, c1 in enumerate(perm):
        c2 = perm[0] if i == len(perm)-1 else perm[i+1]
        distance += euc_2d(cities[c1], cities[c2])
    return distance

def random_permutation(cities):
    perm = [i for i in range(len(cities))]
    for i in range(len(perm)):
        r = random.randint(i, len(perm)-1)
        perm[i], perm[r] = perm[r], perm[i]
    return perm

def stochastic_two_opt(parent):
    perm = parent.copy()
    c1, c2 = random.randint(0, len(perm)-1), random.randint(0, len(perm)-1)
    exclude = [c1]
    exclude.append(len(perm)-1 if c1 == 0 else c1-1)
    exclude.append(0 if c1 == len(perm)-1 else c1+1)
    while c2 in exclude:
        c2 = random.randint(0, len(perm)-1)
    if c2 < c1:
        c1, c2 = c2, c1
    perm[c1:c2] = reversed(perm[c1:c2])
    return perm, [(parent[c1-1], parent[c1]), (parent[c2-1], parent[c2])]

def is_tabu(permutation, tabu_list):
    for i, c1 in enumerate(permutation):
        c2 = permutation[0] if i == len(permutation)-1 else permutation[i+1]
        if (c1, c2) in tabu_list:
            return True
    return False

def generate_candidate(best, tabu_list, cities):
    perm, edges = None, None
    while perm is None or is_tabu(perm, tabu_list):
        perm, edges = stochastic_two_opt(best['vector'])
    candidate = {'vector': perm}
    candidate['cost'] = cost(candidate['vector'], cities)
    return candidate, edges

def search(cities, tabu_list_size, candidate_list_size, max_iter):
    current = {'vector': random_permutation(cities)}
    current['cost'] = cost(current['vector'], cities)
    best = current.copy()
    tabu_list = []
    for i in range(max_iter):
        candidates = [generate_candidate(current, tabu_list, cities) for _ in range(candidate_list_size)]
        candidates.sort(key=lambda x: x[0]['cost'])
        best_candidate = candidates[0][0]
        best_candidate_edges = candidates[0][1]
        if best_candidate['cost'] < current['cost']:
            current = best_candidate.copy()
            if best_candidate['cost'] < best['cost']:
                best = best_candidate.copy()
            for edge in best_candidate_edges:
                tabu_list.append(edge)
            while len(tabu_list) > tabu_list_size:
                tabu_list.pop(0)
        print(f"> iteration {i+1}, best={best['cost']}")
    return best

if __name__ == "__main__":
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
    max_iter = 100
    tabu_list_size = 15
    max_candidates = 50
    
    # execute the algorithm
    best = search(berlin52, tabu_list_size, max_candidates, max_iter)
    print(f"Done. Best Solution: c={best['cost']}, v={best['vector']}")