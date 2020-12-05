import networkx as nx
from parse import read_input_file, write_output_file
from utils import *
import sys
from os.path import basename, normpath
import glob
from itertools import permutations
from mip import *


def solve(G, s, h = 0):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
    Returns:
        D: Dictionary mapping for student to breakout room r e.g. {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    """
    numnodes = G.number_of_nodes()
    #for maxrooms in range(1, numnodes + 1):

    m = Model()
    m.emphasis = SearchEmphasis.FEASIBILITY
    print(m.emphasis)
    #numnodes = 4
    edgesdata = G.edges.data()
    num_edges = len(edgesdata)
    stress_tri = []
    happiness_tri = []
    for i in range(numnodes-1):
        stress_tri.append([j[2]['stress'] for j in edgesdata if j[0] == i])
    print("stress_tri",stress_tri)
    for i in range(numnodes-1):
        happiness_tri.append([j[2]['happiness'] for j in edgesdata if j[0] == i])
    sameroom = [[m.add_var(var_type=BINARY) for j in range(numnodes - i - 1)] for i in range(numnodes-1)]
    #happiness_tri = [[1,2,3],[2,0],[1]]
    print("happiness_tri",happiness_tri)
    print(sum([len(i) for i in happiness_tri]))
    #print(sum([len(i) for i in sameroom]))
    if(h > 0):
        m += xsum(sameroom[i][j]*happiness_tri[i][j] for i in range(numnodes - 1) for j in range(numnodes-i-1)) >= h
    else:
        m.objective = maximize(xsum(sameroom[i][j]*happiness_tri[i][j] for i in range(numnodes - 1) for j in range(numnodes-i-1)))
        #m.objective = maximize(xsum(c[i][j]*y[i][j] for i in range(2) for j in range(2)))
    print("checkpoint1")
    print('model has {} vars, {} constraints and {} nzs'.format(m.num_cols, m.num_rows, m.num_nz))
    m += xsum(sameroom[i][j]*stress_tri[i][j] for i in range(numnodes - 1) for j in range(numnodes-i-1)) <= s
    # THE NEXT LINE IS THE LINE TO ADD CONSTRAINTS IF WE PURSUE SCRAPING
    # m += xsum(sameroom[i][j]*happiness_tri[i][j] for i in range(numnodes - 1) for j in range(numnodes-i-1)) <= 162.796

    
    for i in range(numnodes-2):
        for j in range(i+1,numnodes-1):
            for k in range(j+1,numnodes):
                # print(i,j,k)
                # print(i,j-i-1)
                # print(j,k-j-1)
                # print(i,k-i-1)
                m += sameroom[i][j-i-1] + sameroom[j][k-j-1] <= sameroom[i][k-i-1]  + 1
                m += sameroom[i][j-i-1] + sameroom[i][k-i-1] <= sameroom[j][k-j-1]  + 1
                m += sameroom[i][k-i-1] + sameroom[j][k-j-1] <= sameroom[i][j-i-1]  + 1

    # for i in range(numnodes):
    #     m += xsum(sameroom[i][j] for j in range(numnodes - i - 1)) + xsum(sameroom[j][i-j-1] for j in range(i+1,numnodes-1))
    
    minstress = minimumstress(G)
    maxstress = maximumstress(G)
    minpair = numpairs(G.number_of_nodes(), s, minstress[2]['stress'])
    maxpair = numpairs(G.number_of_nodes(), s, maxstress[2]['stress'])
    min_p = room_student_stress_bound(G.number_of_nodes(),minpair[1],minpair[0],minstress)
    print(min_p)
    max_p = room_student_stress_bound(G.number_of_nodes(),maxpair[1],maxpair[0],maxstress)
    print(max_p)
    m += xsum(sameroom[i][j] for i in range(numnodes - 1) for j in range(numnodes-i-1)) <= min_p
    m += xsum(sameroom[i][j] for i in range(numnodes - 1) for j in range(numnodes-i-1)) >= max_p
    print("checkpoint2")
    m.optimize()

    print(m.num_solutions)
    if m.num_solutions:
        # print(y[0][0].x)
        # print(y[0][1].x)
        # print(y[1][0].x)
        # print(y[1][1].x)
        # room_to_student = []
        # for x in range(10):
        # print("optimum solution: ", x)
        
        stress_sum = 0
        happiness_sum = 0
        roomsmapping = {0:0}
        currroom = 0
        flag = [False for _ in range(numnodes)]
        flag[0] = True
        for i in range(numnodes-1):
            if not flag[i]:
                flag[i] = True
                currroom += 1
                roomsmapping[i] = currroom
            for j in range(numnodes - i - 1):
                if sameroom[i][j].x >= 0.99:
                    flag[j+i+1] = True
                    print(i," ", j + i + 1)
                    roomsmapping[j+i+1] = roomsmapping[i] 
                    stress_sum += stress_tri[i][j]
                    happiness_sum += happiness_tri[i][j]
        rooms = len(set(roomsmapping.values()))
        print("ROOMS:", rooms)
        print("Valid:", is_valid_solution(roomsmapping, G, s, rooms))
        print(roomsmapping)
        print(stress_sum)
        print(happiness_sum)
        
    
    pass

def pairidentifier(i, j, n):
    return i*n + j

def numpairs(n, stressbudget, minimum):
    for k in range(1, n):
        for l in reversed(range(int(n/k), n - k + 2)):
            if stressbudget/k > (minimum * comp(l)):
                return (k, l)

def room_student_stress_bound(n,l,k,stress):
    sum = 0
    while n > 0:
        if(n<l):
            sum += comp(n)
            n = 0
        else:
            sum += comp(l)
            n = n - l
    return sum
    


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
    #nonodes = G.number_of_nodes()
    # for edge in G.edges:
        # print('identifier for', edge, ':\t', pairidentifier(edge[0], edge[1], nonodes))
    #print(is_valid_solution({0: 0, 1: 0, 4: 0, 8: 0, 13: 0, 19: 0, 2: 1, 6: 1, 9: 1, 11: 1, 14: 1, 15: 1, 17: 1, 18: 1, 3: 2, 5: 3, 7: 4, 10: 4, 12: 4, 16: 4}, G, s, 4))
    solve(G,s)
    # numnodes = G.number_of_nodes()
    # edgesdata = G.edges.data()
    # sameroom = [[m.add_var(var_type=BINARY) for i in range(j,numnodes)] for j in range(numnodes)]
    # # n, V = numnodes, set(range(numnodes))
    # # print(V)
    # stress_tri = []
    # for i in range(numnodes):
    #     stress_tri.append([j[2]['stress'] for j in edgesdata if j[0] == i])
    #print(sameroom)
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
    # minpair = numpairs(G.number_of_nodes(), s, minstress[2]['stress'])
    # maxpair = numpairs(G.number_of_nodes(), s, maxstress[2]['stress'])
    # print("min", minpair)
    # print("max", maxpair)
    # print("min with min", room_student_stress_bound(G.number_of_nodes(),minpair[1],minpair[0],minstress))
    # print("max with max",room_student_stress_bound(G.number_of_nodes(),maxpair[1],maxpair[0],maxstress))
    # print("min with max",room_student_stress_bound(G.number_of_nodes(),minpair[1],minpair[0],maxstress))
    # print("max with min",room_student_stress_bound(G.number_of_nodes(),maxpair[1],maxpair[0],minstress))


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
