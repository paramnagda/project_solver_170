import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_happiness
import sys
from os.path import basename, normpath
import glob
from itertools import permutations
from mip import *


def solve(G, s):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
    Returns:
        D: Dictionary mapping for student to breakout room r e.g. {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    """
    m = Model()
    numnodes = G.number_of_nodes()
    edgesdata = G.edges.data()
    sameroom = [m.add_var(var_type=BINARY) for edge in range(len(edgesdata))]
    happinesses = [edge[2]['happiness'] for edge in edgesdata]
    stresses = [edge[2]['stress'] for edge in edgesdata]
    m.objective = maximize(xsum(sameroom[i] * happinesses[i] for i in I))
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                

    pass

def pairidentifier(i, j, n):
    return i*n + j

def numpairs(n, stressbudget, minimum):
    for k in range(1, n):
        for l in reversed(range(int(n/k), n - k + 1)):
            if stressbudget/k > (minimum * comp(l)):
                return (k, l)

def comp(m):
    return m*(m - 1)/2

def minimumstress(G):
    return min(G.edges.data(), key = lambda x: x[2]['stress'])

def maximumstress(G):
    return max(G.edges.data(), key = lambda x: x[2]['stress'])


# def bruteforce(G, s, minrooms, maxrooms):
#     students = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     for i in range(minrooms, maxrooms + 1):
        
#         for i in list(perm):


if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G, s = read_input_file(path)
    nonodes = G.number_of_nodes()
    # for edge in G.edges:
        # print('identifier for', edge, ':\t', pairidentifier(edge[0], edge[1], nonodes))
    m = Model()
    numnodes = G.number_of_nodes()
    edgesdata = G.edges.data()
    n, V = numnodes, set(range(numnodes))
    print(V)
    # sameroom = [m.add_var(var_type=BINARY) for edge in range(len(edgesdata))]
    # happinesses = [edge[2]['happiness'] for edge in edgesdata]
    # stresses = [edge[2]['stress'] for edge in edgesdata]
    # print(len(sameroom))
    # print(len(happinesses))
    # print(happinesses)
    # print(len(stresses))
    # print(stresses)
    # minstress = minimumstress(G)
    # maxstress = maximumstress(G)
    # print("min", numpairs(G.number_of_nodes(), s, minstress[2]['stress']))
    # print("max", numpairs(G.number_of_nodes(), s, maxstress[2]['stress']))

    #D, k = solve(G, s)
    #assert is_valid_solution(D, G, s, k)
    #print("Total Happiness: {}".format(calculate_happiness(D, G)))
    #write_output_file(D, 'outputs/small-1.out')

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G, s = read_input_file(path)
#     D, k = solve(G, s)
#     assert is_valid_solution(D, G, s, k)
#     print("Total Happiness: {}".format(calculate_happiness(D, G)))
#     write_output_file(D, 'outputs/small-1.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob (need to import it)
# if __name__ == '__main__':
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         G, s = read_input_file(input_path)
#         D, k = solve(G, s)
#         assert is_valid_solution(D, G, s, k)
#         happiness = calculate_happiness(D, G)
#         write_output_file(D, output_path)

# if __name__ == '__main__':
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         #output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         G, s = read_input_file(input_path)
#         if s == 95 or s == 92.786 or s == 99.502:
#             print(input_path)
