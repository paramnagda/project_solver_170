import networkx as nx
from parse import read_input_file, write_output_file
from utils import *
import sys
import os.path
from os.path import basename, normpath
import time
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
    valid = False
    stress_sum = 0
    ctr =0
    happiness_sum = float('inf') 
    #for maxrooms in range(1, numnodes + 1):
    ctr = 0
    while (not valid and ctr < 50):
        ctr += 1
        m = Model()
        m.emphasis = SearchEmphasis.FEASIBILITY
        print(m.emphasis)
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
        m += xsum(sameroom[i][j]*happiness_tri[i][j] for i in range(numnodes - 1) for j in range(numnodes-i-1)) <= happiness_sum - happiness_sum * 0.001
        # m += xsum(sameroom[i][j]*happiness_tri[i][j] for i in range(numnodes - 1) for j in range(numnodes-i-1)) <= <INSERT VALUE TO SCRAPE HERE>  
        
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

        for i in range(numnodes):
            m += xsum(sameroom[i][j] for j in range(numnodes - i - 1)) + xsum(sameroom[j][i-j-1] for j in range(i-1)) >= 1
        
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
        ctr += 1
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
            for i in range(numnodes):
                if not flag[i]:
                    flag[i] = True
                    currroom += 1
                    roomsmapping[i] = currroom
                for j in range(numnodes - i - 1):
                    if sameroom[i][j].x >= 0.99:
                        flag[j+i+1] = True
                        #print(i," ", j + i + 1)
                        roomsmapping[j+i+1] = roomsmapping[i] 
                        stress_sum += stress_tri[i][j]
                        happiness_sum += happiness_tri[i][j]
            rooms = len(set(roomsmapping.values()))
            valid = is_valid_solution(roomsmapping, G, s, rooms)
            print("ROOMS:", rooms)
            print("budget: ", s)
            print("Valid:", valid)
            print(roomsmapping)
            print(stress_sum)
            print(happiness_sum)
            if valid:
                print("VALIDDDDD")
                print("Iterations of ILP taken: ", ctr)
                return roomsmapping, rooms
        else:
            print("no valid solution")
            print("failed budget: ", s)
            return 0, 0
    return 0, 0

def solverNotOptimal(G, s):
    numnodes = G.number_of_nodes()
    valid = False
    estimatedrooms = numnodes/2
    sperroom = s/estimatedrooms
    pendingstudents = list(G.nodes())
    curroom = 0
    roomsmapping = {}
    while pendingstudents:
        student = pendingstudents[0]
        roomsmapping[student] = curroom
        curroom += 1
        pendingstudents.remove(student)
        print(student)
        print(pendingstudents)
        potentialpartners = [partner for partner in pendingstudents if G[student][partner]["stress"] <= sperroom]
        if potentialpartners:
            partner = max(potentialpartners, key = lambda x: G[student][x]["happiness"])
            roomsmapping[partner] = roomsmapping[student]
            pendingstudents.remove(partner)
    rooms = len(set(roomsmapping.values()))
    valid = is_valid_solution(roomsmapping, G, s, rooms)
    print("ROOMS:", rooms)
    print("budget: ", s)
    print("Valid:", valid)
    print(roomsmapping)
    if valid:
        print("VALIDDDDD")
        return roomsmapping, rooms
    else:
        print("INVALID: Could not find appropriate matching")
        return 0, 0

def solverNotOptimal3students(G, s):
    numnodes = G.number_of_nodes()
    valid = False
    estimatedrooms = int(numnodes/3)
    sperroom = s/estimatedrooms
    pendingstudents = list(G.nodes())
    curroom = 0
    roomsmapping = {}
    while pendingstudents:
        student = pendingstudents[0]
        roomsmapping[student] = curroom
        curroom += 1
        pendingstudents.remove(student)
        print(student)
        print(pendingstudents)
        potentialpartners = [partner for partner in pendingstudents if G[student][partner]["stress"] <= sperroom]
        if potentialpartners:
            partner = max(potentialpartners, key = lambda x: G[student][x]["happiness"])
            roomsmapping[partner] = roomsmapping[student]
            stressleft = sperroom - G[student][partner]["stress"]
            pendingstudents.remove(partner)
            potentialpartners2 = [partner2 for partner2 in pendingstudents if G[student][partner2]["stress"] + G[partner][partner2]["stress"] <= stressleft]
            if potentialpartners2 and stressleft:
                partner2 = max(potentialpartners2, key = lambda x: G[student][x]["happiness"] + G[partner][x]["happiness"])
                roomsmapping[partner2] = roomsmapping[student]
                pendingstudents.remove(partner2)
    rooms = len(set(roomsmapping.values()))
    valid = is_valid_solution(roomsmapping, G, s, rooms)
    print("ROOMS:", rooms)
    print("budget: ", s)
    print("Valid:", valid)
    print(roomsmapping)
    if valid:
        print("VALIDDDDD")
        return roomsmapping, rooms
    else:
        print("INVALID: Could not find appropriate matching")
        return 0, 0

def solverNotOptimal4students(G, s):
    numnodes = G.number_of_nodes()
    valid = False
    estimatedstudentsperroom = 5
    estimatedrooms = int(numnodes/estimatedstudentsperroom)
    sperroom = s/estimatedrooms
    pendingstudents = list(G.nodes())
    curroom = 0
    roomsmapping = {}
    while pendingstudents:
        student = pendingstudents[0]
        roomsmapping[student] = curroom
        curroom += 1
        pendingstudents.remove(student)
        print(student)
        print(pendingstudents)
        potentialpartners = [partner for partner in pendingstudents if G[student][partner]["stress"] <= sperroom]
        if potentialpartners:
            partner = max(potentialpartners, key = lambda x: G[student][x]["happiness"])
            roomsmapping[partner] = roomsmapping[student]
            pendingstudents.remove(partner)
            stressleft = sperroom - G[student][partner]["stress"]
            potentialpartners2 = [partner2 for partner2 in pendingstudents if G[student][partner2]["stress"] + G[partner][partner2]["stress"] <= stressleft]
            if potentialpartners2 and stressleft:
                partner2 = max(potentialpartners2, key = lambda x: G[student][x]["happiness"] + G[partner][x]["happiness"])
                roomsmapping[partner2] = roomsmapping[student]
                pendingstudents.remove(partner2)
                stressleft = stressleft - (G[student][partner2]["stress"] + G[partner][partner2]["stress"])
                potentialpartners3 = [partner3 for partner3 in pendingstudents if stress4(G, student, partner, partner2, partner3) <= sperroom]
                if potentialpartners3 and stressleft:
                    partner3 = max(potentialpartners3, key = lambda x: happiness4(G, student, partner, partner2, x))
                    roomsmapping[partner3] = roomsmapping[student]
                    pendingstudents.remove(partner3)
                    stressleft = stressleft - stress4(G, student, partner, partner2, partner3)
                    potentialpartners4 = [partner4 for partner4 in pendingstudents if stress5(G, student, partner, partner2, partner3, partner4) <= sperroom]
                    if potentialpartners4 and stressleft:
                        partner4 = max(potentialpartners4, key = lambda x: happiness5(G, student, partner, partner2, partner3, x))
                        roomsmapping[partner4] = roomsmapping[student]
                        pendingstudents.remove(partner4)
                    
    rooms = len(set(roomsmapping.values()))
    valid = is_valid_solution(roomsmapping, G, s, rooms)
    print("ROOMS:", rooms)
    print("budget: ", s)
    print("Valid:", valid)
    print(roomsmapping)
    if valid:
        print("VALIDDDDD")
        return roomsmapping, rooms
    else:
        print("INVALID: Could not find appropriate matching")
        return 0, 0

def solverNotOptimalkstudents(G, s, k):
    numnodes = G.number_of_nodes()
    valid = False
    estimatedstudentsperroom = k
    estimatedrooms = int(numnodes/estimatedstudentsperroom)
    sperroom = s/estimatedrooms
    pendingstudents = list(G.nodes())
    curroom = 0
    roomsmapping = {}
    while pendingstudents:
        student = pendingstudents[0]
        roomsmapping[student] = curroom
        curroom += 1
        pendingstudents.remove(student)
        print(student)
        print(pendingstudents)
        thisroom = [student]
        for i in range(k):
            potentialpartners = [partner for partner in pendingstudents if calculate_stress_for_room(thisroom + [partner], G) <= sperroom]
            if potentialpartners:
                partner = max(potentialpartners, key = lambda x: calculate_happiness_for_room(thisroom + [x], G))
                roomsmapping[partner] = roomsmapping[student]
                pendingstudents.remove(partner)
            else:
                break
        
                    
    rooms = len(set(roomsmapping.values()))
    valid = is_valid_solution(roomsmapping, G, s, rooms)
    print("ROOMS:", rooms)
    print("budget: ", s)
    print("Valid:", valid)
    print(roomsmapping)
    if valid:
        print("VALIDDDDD")
        return roomsmapping, rooms
    else:
        print("INVALID: Could not find appropriate matching")
        return 0, 0

def stress4(G, s0, s1, s2, s3):
    s = [s0,s1,s2,s3]
    s.sort()
    #print("stress", s)
    return G[s[0]][s[1]]["stress"] + G[s[0]][s[2]]["stress"] + G[s[0]][s[3]]["stress"] + G[s[1]][s[2]]["stress"] + G[s[1]][s[3]]["stress"] + G[s[2]][s[3]]["stress"] 

def happiness4(G, s0, s1, s2, s3):
    s = [s0,s1,s2,s3]
    s.sort()
    #print("happiness", s)
    return G[s[0]][s[1]]["happiness"] + G[s[0]][s[2]]["happiness"] + G[s[0]][s[3]]["happiness"] + G[s[1]][s[2]]["happiness"] + G[s[1]][s[3]]["happiness"] + G[s[2]][s[3]]["happiness"] 
    
def stress5(G, s0, s1, s2, s3, s4):
    s = [s0,s1,s2,s3,s4]
    s.sort()
    #print("stress", s)
    return G[s[0]][s[1]]["stress"] + G[s[0]][s[2]]["stress"] + G[s[0]][s[3]]["stress"] + G[s[0]][s[4]]["stress"] + G[s[1]][s[2]]["stress"] + G[s[1]][s[3]]["stress"] + G[s[1]][s[4]]["stress"] + G[s[2]][s[3]]["stress"] + G[s[2]][s[4]]["stress"] + G[s[3]][s[4]]["stress"]  

def happiness5(G, s0, s1, s2, s3, s4):
    s = [s0,s1,s2,s3,s4]
    s.sort()
    #print("happiness", s)
    return G[s[0]][s[1]]["happiness"] + G[s[0]][s[2]]["happiness"] + G[s[0]][s[3]]["happiness"] + G[s[0]][s[4]]["happiness"] + G[s[1]][s[2]]["happiness"] + G[s[1]][s[3]]["happiness"] + G[s[1]][s[4]]["happiness"] + G[s[2]][s[3]]["happiness"] + G[s[2]][s[4]]["happiness"] + G[s[3]][s[4]]["happiness"]  

def trivialSolver(G, s):
    roomsmapping = {}
    for student in list(G.nodes()):
        roomsmapping[student] = student
    rooms = len(set(roomsmapping.values()))
    return roomsmapping, rooms        

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

def writeupdate(solved, unsolved, skippedcozexists):
    with open("update.txt", "w") as fo:
        fo.write("**solved in this iteration:**\n")
        for key, value in solved.items():
            fo.write(str(key) + " " + str(value) + "\n")
        fo.write("**skipped in this iteration:**\n")
        for key, value in unsolved.items():
            fo.write(str(key) + " " + str(value) + "\n")
        fo.write("**skipped coz exists:**\n")        
        for skipped in skippedcozexists:
            fo.write(skipped + "\n")
        fo.write("**total solved in this iteration**" + str(len(solved.keys())) + "\n")
        fo.write("**trivially solved in this iteration**" + str(len(unsolved.keys())) + "\n")   
        fo.write("**total skipped coz exists in this iteration**" + str(len(skippedcozexists)) + "\n")
   
        fo.close()


if __name__ == '__main__':
    start = time.time()
    ctr = 0
    unsolvedctr = 0
    if (len(sys.argv) == 2):
        path = sys.argv[1]
        G, s = read_input_file(path)
        output_path = 'naivekstudentsrandomtesting3/' + basename(normpath(path))[:-3] + '.out'
        D, k = solverNotOptimalkstudents(G, s, 10)
        #D, k = solverNotOptimal4students(G, s)
        if D or k:
            assert is_valid_solution(D, G, s, k)
            happiness = calculate_happiness(D, G)
            print("happiness: ", calculate_happiness(D, G))
            write_output_file(D, output_path)
        else:
            print("NO SOLUTION FOUND FOR", path)
    else:
        unsolveddict = {}
        happinessdict = {}
        skippedcozexists = []
        inputs = glob.glob('inputs/*')
        for input_path in inputs:
            if "medium" in input_path or "large" in input_path:
                output_path = 'naivekstudentsrandomtesting3/' + basename(normpath(input_path))[:-3] + '.out'
                if os.path.isfile(output_path):
                    ctr += 1
                    print("skipping", output_path, "because it already exists")
                    skippedcozexists += [output_path]
                    writeupdate(happinessdict, unsolveddict, skippedcozexists)
                    continue
                else:
                    print("now attempting", input_path)
                    G, s = read_input_file(input_path)
                    try:
                        #D, k = solverNotOptimal4students(G, s)
                        D, k = solverNotOptimalkstudents(G, s, 12)
                        if D or k:
                            assert is_valid_solution(D, G, s, k)
                            ctr += 1
                            happiness = calculate_happiness(D, G)
                            write_output_file(D, output_path)
                            happinessdict[input_path] = happiness
                        else:
                            # D, k = trivialSolver(G, s)
                            # assert is_valid_solution(D, G, s, k)
                            # write_output_file(D, output_path)
                            unsolveddict[input_path] = "did not give valid solution so trivially solved"
                            unsolvedctr += 1
                        print("solved list", happinessdict)
                        print("unsolved list", unsolveddict)
                        writeupdate(happinessdict, unsolveddict, skippedcozexists)
                    except Exception:
                        # D, k = trivialSolver(G, s)
                        # assert is_valid_solution(D, G, s, k)
                        # write_output_file(D, output_path)
                        unsolveddict[input_path] = "gave error so trivially solved"
                        unsolvedctr += 1
                        writeupdate(happinessdict, unsolveddict, skippedcozexists)
                        continue
    end = time.time()
    print("total time taken:", end - start)
    print("total solved inputs:", ctr)
    print("total unsolved so trivially solved inputs:", unsolvedctr)
    


        
    # assert len(sys.argv) == 2
    # path = sys.argv[1]
    # G, s = read_input_file(path)
    # output_path = 'outputs/' + basename(normpath(path))[:-3] + '.out'
    # D, k = solve(G, s)
    # if D or k:
    #     assert is_valid_solution(D, G, s, k)
    #     happiness = calculate_happiness(D, G)
    #     write_output_file(D, output_path)
    # else:
    #     print("NO SOLUTION FOUND FOR", path)
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

    # for input_path in inputs:
    #     if "large" in input_path or "medium" in input_path or "small" in input_path:
    #         output_path = 'naiveoutputs/' + basename(normpath(input_path))[:-3] + '.out'
    #         if os.path.isfile(output_path):
    #             ctr += 1
    #             continue
    #             print("skipping", output_path, "because it already exists")
    #             skippedcozexists += [output_path]
    #             writeupdate(happinessdict, unsolveddict, skippedcozexists)
    #             continue
    #         else:
    #             unsolvedctr += 1
    #             continue


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
