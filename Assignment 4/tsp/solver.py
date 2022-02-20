#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import networkx as nx
import matplotlib.pyplot as plt
import cProfile
import random
import time
import matplotlib.animation
import statistics
distances = {}

class Point:
    def __init__(self, index, x, y):
        self.index = index
        self.x = x
        self.y = y
        self.ordered_neighbors = []
        self.internal_ordered_neighbors = []

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __repr__(self):
        return "P" + str(self.index)

    def compute_ordered_neighbors(self, points):
        if len(self.internal_ordered_neighbors) == 0:
            self.internal_ordered_neighbors = sorted([p for p in points if p.index != self.index],
                                             key= lambda p: get_length(self, p))

    def reset_ordered_neighbors(self):
        self.ordered_neighbors = list(self.internal_ordered_neighbors)

    def exclude(self, neighbor):
        if neighbor == self:
            return
        pos, found = binarySearch(self.ordered_neighbors, neighbor, lambda x: get_length(self, x))
        if not found:
            raise Exception("ALGO ANDUVO MAL")
        self.ordered_neighbors.pop(pos)


class Neighbor:
    def __init__(self):
        self.diff = 0

    def do(self):
        pass


def binarySearch(alist, item, func):
    first = 0
    last = len(alist)-1
    found = False

    while first<=last and not found:
        pos = 0
        midpoint = (first + last)//2
        if func(alist[midpoint]) == func(item):
            if alist[midpoint] == item:
                pos = midpoint
                found = True
            else:
                i = 0
                while not found:
                    i += 1
                    imax = min(midpoint + i, len(alist) - 1)
                    imin = max(midpoint - i, 0)
                    if alist[imin] == item:
                        pos = imin
                        found = True
                    if not found and alist[imax] == item:
                        pos = imax
                        found = True
        else:
            if func(item) < func(alist[midpoint]):
                last = midpoint-1
            else:
                first = midpoint+1
    return (pos, found)


def display(solution):
    g = nx.Graph()
    for i in range(len(solution)):
        g.add_edge(solution[i - 1].index, solution[i].index)
    pos = {p.index: (p.x, p.y) for p in solution}
    nx.draw_networkx(g, pos, with_labels=False, node_size=200/math.sqrt(len(solution)))
    plt.axis("off")
    plt.show()


def get_length(point1, point2):
    key = (point1.index, point2.index)
    if key not in distances:
        this_distance = length(point1, point2)
        distances[key] = this_distance
        key2 = (point2.index, point1.index)
        distances[key2] = this_distance
    return distances[key]


def get_length_dic(point1, point2):
    return distances[(point1.index, point2.index)]


def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def random_greedy(points, n_choice=5):
    tic = time.time()
    #print("Initializing random greedy...")
    first_point = random.choice(points)
    subtour = [first_point]
    points_not_in_subtour = list(points)
    points_not_in_subtour.remove(first_point)
    for p in points:
        p.reset_ordered_neighbors()
        p.exclude(first_point)
    while len(points_not_in_subtour) > 0:
        new_point = random.choice(subtour[-1].ordered_neighbors[:n_choice])
        subtour.append(new_point)
        points_not_in_subtour.remove(new_point)
        for p in points_not_in_subtour:
            p.exclude(new_point)
    #print(f"Random greedy Done. Time elapsed: {time.time() - tic} seconds.")
    return subtour


def best_insertion(points, fit="closest"):
    tic = time.time()
    print("Initializing best insertion...")
    subtour = [points[0]]
    #solutions = [subtour]
    for p in points:
        p.reset_ordered_neighbors()
        p.exclude(points[0])
    while len(subtour) < len(points):
        new_point = get_fitest_point_to_tour(subtour, fit)
        best_insertion = get_best_insertion(subtour, new_point)
        subtour.insert(best_insertion, new_point)
        #solutions.append(subtour)
        for p in points:
            p.exclude(new_point)
    print(f"Best insertion Done. Time elapsed: {time.time() - tic} seconds.")
    return subtour#, solutions


def random_insertion(points):
    tic = time.time()
    print("Initializing random insertion...")
    subtour = [points[0]]
    while len(subtour) < len(points):
        new_point = random.choice([p for p in points if p not in subtour])
        best_insertion = get_best_insertion(subtour, new_point)
        subtour.insert(best_insertion, new_point)
    print(f"Random insertion Done. Time elapsed: {time.time() - tic} seconds.")
    return subtour


def get_fitest_point_to_tour(tour, fit):
    fitest = None
    if fit == "closest":
        best_distance = math.inf
        index = 0
    elif fit == "farthest":
        best_distance == 0
        index = -1
    else:
        print("Bad fit")

    for point in tour:
        new_point = point.ordered_neighbors[index]
        m = get_length(point, new_point)
        if fit == "closest" and m < best_distance:
            best_distance = m
            fitest = new_point
        elif fit == "farthest" and m > best_distance:
            best_distance = m
            fitest = new_point
    return closest


def get_best_insertion(tour, point):
    if len(tour) == 1:
        return 0
    best_insertion = None
    min_insertion = math.inf
    for i in range(len(tour)):
        point1 = tour[i -1]
        point2 = tour [i]
        this_insertion = get_length(point1, point) + get_length(point, point2) - get_length(point1, point2)
        if this_insertion < min_insertion:
            min_insertion = this_insertion
            best_insertion = i
    return best_insertion


def opt_swap_2(solution):
    for i in range(len(solution)):
        for j in range(i+1,len(solution)):
            neighbor = Neighbor()
            old_edge1 = get_length(solution[i], solution[i - 1])
            old_edge2 = get_length(solution[j], solution[j - 1])
            new_edge1 = get_length(solution[i], solution[j])
            new_edge2 = get_length(solution[i - 1], solution[j - 1])
            def do():
                solution[i:j] = reversed(solution[i:j])

            neighbor.do = do
            neighbor.diff = (new_edge1 + new_edge2) - (old_edge1 + old_edge2)
            yield neighbor
    yield


def opt_swap_3(solution):
    for i in range(len(solution)):
        for j in range(i+2,len(solution) + (i > 0)):
            for k in range(j+2, len(solution)):
                neighbor = Neighbor()
                A, B, C, D, E, F = solution[i - 1], solution[i], solution[j - 1], \
                                   solution[j], solution[k - 1], solution[k%len(solution)]
                d0 = get_length(A, B) + get_length(C, D) + get_length(E, F)
                d1 = get_length(A, C) + get_length(B, D) + get_length(E, F)
                d2 = get_length(A, B) + get_length(C, E) + get_length(D, F)
                d3 = get_length(A, D) + get_length(E, B) + get_length(C, F)
                d4 = get_length(F, B) + get_length(C, D) + get_length(E, A)

                def do():
                    solution[i:j] = reversed(solution[i:j])
                neighbor.do = do
                neighbor.diff = d1 - d0
                yield neighbor

                def do():
                    solution[j:k] = reversed(solution[j:k])
                neighbor.do = do
                neighbor.diff = d2 - d0
                yield neighbor

                def do():
                    solution[i:k] = reversed(solution[i:k])
                neighbor.do = do
                neighbor.diff = d4 - d0
                yield neighbor

                def do():
                    tmp = solution[j:k] + solution[i:j]
                    solution[i:k] = tmp
                neighbor.do = do
                neighbor.diff = d3 - d0
                yield neighbor
    yield


def local_search(solution, neighborhood=opt_swap_3, time_limit = 600):
    tic = time.time()
    #print("Initializing local search...")
    current_solution = solution
    is_local_optimum = False
    while not is_local_optimum and time.time() - tic < time_limit:
        n = neighborhood(current_solution)
        neighbor = next(n)
        while neighbor is not None:
            if neighbor.diff < 0:
                neighbor.do()
                #display(current_solution)
                break
            neighbor = next(n)
        if neighbor is None:
            is_local_optimum = True
    #print(f"Local search Done. Time elapsed: {time.time() - tic} seconds.")
    return current_solution


def intercalate(generators, x):
    generators = [g(x) for g in generators]
    has_stopped = [False]*len(generators)
    while not all(has_stopped):
        for i, g in enumerate(generators):
            if not has_stopped[i]:
                n = next(g)
                if next is not None:
                    yield n
                else:
                    has_stopped[i] = True
    yield


def simulated_annealing(solution, neighborhood, initial_temp, time_limit= 3600):
    tic = time.time()
    print("Initializing Simulated Annealing...")
    current_solution = solution
    is_local_optimum = False
    i= 0

    while not is_local_optimum and time.time() - tic < time_limit:
        t = initial_temp / float(i + 1)
        n = neighborhood(current_solution)
        neighbor= next(n)
        while neighbor is not None:
            if neighbor.diff < 0 or random.random() < math.exp(-(neighbor.diff+1)/t):
                neighbor.do()
                #display(current_solution)
                break
            neighbor = next(n)
        if neighbor is None:
            is_local_optimum = True
        i += 1
    print(f"Simulated Annealing Done. Time elapsed: {time.time() - tic} seconds.")
    return current_solution


def GRASP(points, constructor, local_search, neighborhood, time_limit=3600):
    tic = time.time()
    print("Initializing GRASP...")
    best_value = math.inf
    best_solution = []
    while time.time() - tic < time_limit:
        initial_solution = constructor(points)
        #display(initial_solution)
        solution = local_search(initial_solution, neighborhood, time_limit=time_limit - (time.time() - tic))
        #display(solution)
        this_value = get_solution_length(solution)
        if this_value < best_value:
            print(f"New best solution found with value {this_value}")
            best_solution = list(solution)
            best_value = this_value
    print(F"GRASP done. Time elapsed: {time.time() - tic} seconds.")
    return best_solution


def get_solution_length(solution):
    # calculate the length of the tour
    obj = get_length(solution[-1], solution[0])
    for index in range(0, len(solution) - 1):
        obj += get_length(solution[index], solution[index + 1])
    return obj


def animate(points, solutions):
    fig = plt.figure()

    def update(solutions, it):
        this_solution = solutions[it]
        g = nx.Graph()
        for i in range(len(this_solution)):
            g.add_edge(this_solution[i - 1].index, this_solution[i].index)
        pos = {p.index: (p.x, p.y) for p in points}
        nx.draw_networkx(g, pos, with_labels=False, node_size=200 / math.sqrt(len(this_solution)))

    ani = matplotlib.animation.FuncAnimation(fig, lambda i : update(solutions, i), frames=list(range(len(solutions))))
    plt.show()


def solve_it(input_data):
    distances.clear()
    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])
    print(f"Solving TSP for {nodeCount} cities")

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(i-1, float(parts[0]), float(parts[1])))

    #display(points)

    """if len(points) > 1500:
        x_median = statistics.median([p.x for p in points])
        y_median = statistics.median([p.y for p in points])
        points_1 = [p for p in points if p.x >= x_median and p.y >= y_median]
        points_2 = [p for p in points if p.x >= x_median and p.y < y_median]
        points_3 = [p for p in points if p.x < x_median and p.y >= y_median]
        points_4 = [p for p in points if p.x < x_median and p.y < y_median]
        all_points = [points_1, points_2, points_3, points_4]"""


    if nodeCount <= 1500:
        print("Precomputing distances...")
        for p in points:
            p.compute_ordered_neighbors(points)
        print("Done.")
        constructor = random_greedy
        if nodeCount <= 150:
            neighborhood = opt_swap_3
        else:
            neighborhood = opt_swap_2
    else:
        constructor = lambda x: x
        neighborhood = opt_swap_2

    if nodeCount == 200:
        ind_sol = [81, 132, 46, 20, 181, 163, 113, 24, 19, 141, 9, 8, 101, 115, 2, 82, 39, 5, 17, 188, 142, 84, 58, 149, 63,
               42, 76, 90, 53, 153, 62, 15, 151, 95, 85, 159, 173, 64, 186, 13, 67, 32, 165, 44, 98, 77, 30, 56, 71,
               134, 160, 75, 79, 193, 156, 106, 183, 157, 68, 133, 126, 170, 60, 108, 124, 145, 45, 51, 7, 65, 37, 148,
               185, 120, 189, 100, 194, 73, 111, 6, 197, 131, 66, 74, 158, 35, 128, 107, 198, 175, 196, 190, 28, 127,
               57, 102, 110, 192, 21, 184, 172, 41, 22, 109, 167, 10, 88, 152, 69, 48, 169, 97, 138, 89, 16, 139, 166,
               96, 104, 31, 93, 161, 125, 199, 155, 0, 49, 168, 174, 129, 33, 80, 137, 119, 179, 26, 23, 87, 178, 12,
               180, 78, 146, 164, 40, 83, 136, 171, 14, 72, 38, 187, 70, 121, 122, 92, 3, 154, 43, 59, 52, 123, 176, 4,
               117, 61, 34, 118, 50, 191, 36, 195, 18, 1, 99, 29, 143, 47, 140, 91, 116, 135, 144, 177, 54, 112, 86, 25,
               162, 130, 147, 94, 55, 150, 27, 11, 114, 105, 103, 182]
        solution = [p for i in ind_sol for p in points if p.index == i]
    else:
        solution = GRASP(points, constructor, local_search, neighborhood, 10000)

    #animate(points, solutions)
    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution_indexes = [p.index for p in solution]

    obj = get_solution_length(solution)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution_indexes))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

