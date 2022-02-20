#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import time
import random
distances = {}


class Customer:
    def __init__(self, index, demand, x, y):
        self.index = index
        self.demand = demand
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

    def compute_ordered_neighbors(self, customers):
        if len(self.internal_ordered_neighbors) == 0:
            self.internal_ordered_neighbors = sorted([p for p in customers if p.index != self.index],
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


class Neighbor:
    def __init__(self, name):
        self.diff = 0
        self.name = name

    def do(self):
        pass


class Route:
    def __init__(self, depot, capacity, stop_sequence=[]):
        self.capacity = capacity
        self.remaining_capacity = capacity
        if len(stop_sequence) == 0:
            self.stop_sequence = [depot, depot]
        else:
            self.stop_sequence = stop_sequence

    def is_feasible(self):
        return sum([stop.demand for stop in self.stop_sequence]) <= self.vehicle.capacity

    def get_total_length(self):
        total_length = 0
        for i in range(len(self.stop_sequence) - 1):
            total_length += get_length(self.stop_sequence[i], self.stop_sequence[i + 1])
        return total_length

    def remove_stop_sequence_at(self, index):
        customer = self.stop_sequence.pop(index)
        self.remaining_capacity += customer.demand

    def add_stop_sequence_at(self, index, customer):
        self.stop_sequence.insert(index, customer)
        self.remaining_capacity -= customer.demand

    def add_stop_sequence(self, customer):
        if customer.demand > self.remaining_capacity:
            return False
        self.add_stop_sequence_at(-1, customer)

    def insert_stop_sequence(self, customer):
        if customer.demand > self.remaining_capacity:
            return False
        index = get_best_insertion(self.stop_sequence, customer)
        self.add_stop_sequence_at(index, customer)

    def get_customers(self):
        return self.stop_sequence[1:-1]

    def get_last(self):
        return self.stop_sequence[-2]

    def get_first(self):
        return self.stop_sequence[1]

    def is_empty(self):
        return len(self.stop_sequence) == 2

    def sanitize(self):
        if self.is_empty():
            return
        if self.get_first().index > self.get_last().index:
            self.stop_sequence.reverse()


def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)


def get_length(point1, point2):
    key = (point1.index, point2.index)
    if key not in distances:
        this_distance = length(point1, point2)
        distances[key] = this_distance
        key2 = (point2.index, point1.index)
        distances[key2] = this_distance
    return distances[key]


def random_greedy(depot, customers, n_vehicles, capacity, n_choice=5):
    tic = time.time()
    #print("Initializing random greedy...")
    solution = []
    #while len(solution) == 0:
    for c in customers:
        c.reset_ordered_neighbors()
    depot.reset_ordered_neighbors()
    not_visited = set(customers)
    for i in range(n_vehicles):
        route = Route(depot, capacity)
        small_enough_customers = set(not_visited)
        while len(small_enough_customers) > 0:
            new_customer = random.choice([c for c in route.get_last().ordered_neighbors
                                          if c in small_enough_customers][:n_choice])
            route.add_stop_sequence(new_customer)
            not_visited.remove(new_customer)
            for p in not_visited:
                p.exclude(new_customer)
            small_enough_customers = set([c for c in not_visited if c.demand <= route.remaining_capacity])
        solution.append(route)
        """if len(customers) != len([c.index for r in solution for c in r.get_customers()]):
            print("Retrying construction...")
            solution = []"""

    #print(f"Random greedy Done. Time elapsed: {time.time() - tic} seconds.")
    return solution


def best_insertion(depot, customers, fit="closest"):
    tic = time.time()
    print("Initializing best insertion...")
    solution = []
    for p in points:
        p.reset_ordered_neighbors()
        p.exclude(customers[0])
    while len(subtour) < len(customers):
        new_point = get_fitest_point_to_tour(subtour, fit)
        best_insertion = get_best_insertion(subtour, new_point)
        subtour.insert(best_insertion, new_point)
        #solutions.append(subtour)
        for p in customers:
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
    return fitest


def get_best_insertion(tour, point):
    if len(tour) <= 2:
        return 1#, 2 * get_length(point, tour[0])
    best_insertion = None
    min_insertion = math.inf
    for i in range(1, len(tour)):
        point1 = tour[i - 1]
        point2 = tour [i]
        this_insertion = get_length(point1, point) + get_length(point, point2) - get_length(point1, point2)
        if this_insertion < min_insertion:
            min_insertion = this_insertion
            best_insertion = i
    return best_insertion#, min_insertion


def get_distances_with_insertion(route, out_index, in_customer):
    tour = list(route.stop_sequence)
    out_customer_before, out_customer, out_customer_after = tour[out_index - 1], tour[out_index], tour[out_index + 1]
    old_d = get_length(out_customer_before, out_customer) + get_length(out_customer, out_customer_after)
    tour.pop(out_index)
    in_index = get_best_insertion(tour, in_customer)
    insertion_before = tour[in_index - 1]
    insertion_after = tour[in_index]
    new_d = get_length(insertion_before, in_customer) + get_length(in_customer, insertion_after) - \
            get_length(insertion_before, insertion_after) + get_length(out_customer_before, out_customer_after)

    return in_index, old_d, new_d


def opt_swap_2_neighborhood(solution, customers):
    for route in solution:
        if not route.is_empty():
            for i in range(1, len(route.stop_sequence) - 2):
                for j in range(i + 1,len(route.stop_sequence) - 1):
                    neighbor = Neighbor("swap2")
                    old_edge1 = get_length(route.stop_sequence[i], route.stop_sequence[i - 1])
                    old_edge2 = get_length(route.stop_sequence[j], route.stop_sequence[j - 1])
                    new_edge1 = get_length(route.stop_sequence[i], route.stop_sequence[j])
                    new_edge2 = get_length(route.stop_sequence[i - 1], route.stop_sequence[j - 1])
                    def do():
                        route.stop_sequence[i:j] = reversed(route.stop_sequence[i:j])
                        #print("swap2 done")

                    neighbor.do = do
                    neighbor.diff = (new_edge1 + new_edge2) - (old_edge1 + old_edge2)
                    yield neighbor
    yield


def opt_swap_3_neighborhood(solution, customers):
    for route in solution:
        for i in range(1, len(route.stop_sequence) - 1):
            for j in range(i+2,len(route.stop_sequence) - 1):
                for k in range(j+2, len(route.stop_sequence) - 1):
                    neighbor = Neighbor("swap3")
                    A, B, C, D, E, F = route.stop_sequence[i - 1], route.stop_sequence[i], route.stop_sequence[j - 1], \
                                       route.stop_sequence[j], route.stop_sequence[k - 1], route.stop_sequence[k%len(route.stop_sequence)]
                    d0 = get_length(A, B) + get_length(C, D) + get_length(E, F)
                    d1 = get_length(A, C) + get_length(B, D) + get_length(E, F)
                    d2 = get_length(A, B) + get_length(C, E) + get_length(D, F)
                    d3 = get_length(A, D) + get_length(E, B) + get_length(C, F)
                    d4 = get_length(F, B) + get_length(C, D) + get_length(E, A)

                    def do():
                        route.stop_sequence[i:j] = reversed(route.stop_sequence[i:j])
                        #print("swap31 done")
                    neighbor.do = do
                    neighbor.diff = d1 - d0
                    yield neighbor

                    def do():
                        route.stop_sequence[j:k] = reversed(route.stop_sequence[j:k])
                        #print("swap32 done")
                    neighbor.do = do
                    neighbor.diff = d2 - d0
                    yield neighbor

                    def do():
                        route.stop_sequence[i:k] = reversed(route.stop_sequence[i:k])
                        #print("swap33 done")
                    neighbor.do = do
                    neighbor.diff = d4 - d0
                    yield neighbor

                    def do():
                        tmp = route.stop_sequence[j:k] + route.stop_sequence[i:j]
                        route.stop_sequence[i:k] = tmp
                        #print("swap34 done")
                    neighbor.do = do
                    neighbor.diff = d3 - d0
                    yield neighbor
    yield


def exchange_neighborhood(solution, customers):
    for i in range(len(solution) - 1):
        for j in range(i + 1, len(solution)):
            route1 = solution[i]
            route2 = solution[j]
            for c1 in range(1, len(route1.stop_sequence) - 1):
                for c2 in range(1, len(route2.stop_sequence) - 1):
                    customer1 = route1.stop_sequence[c1]
                    customer2 = route2.stop_sequence[c2]
                    available1 = route1.remaining_capacity + customer1.demand
                    available2 = route2.remaining_capacity + customer2.demand
                    if customer2.demand <= available1 and customer1.demand <= available2:
                        in_index2, old_d1, new_d1 = get_distances_with_insertion(route1, out_index=c1, in_customer=customer2)
                        in_index1, old_d2, new_d2 = get_distances_with_insertion(route2, out_index=c2, in_customer=customer1)
                        """before_customer1, after_customer1 = route1.stop_sequence[c1 - 1], route1.stop_sequence[c1 + 1]
                        before_customer2, after_customer2 = route2.stop_sequence[c2 - 1], route2.stop_sequence[c2 + 1]
                        old_d = get_length(before_customer1, customer1) + get_length(customer1, after_customer1) + \
                                get_length(before_customer2, customer2) + get_length(customer2, after_customer2)
                        new_d = get_length(before_customer1, customer2) + get_length(customer2, after_customer1) + \
                                get_length(before_customer2, customer1) + get_length(customer1, after_customer2)"""
                        neighbor = Neighbor("exchange")
                        neighbor.diff = new_d1 + new_d2 - old_d1 - old_d2
                        def do():
                            route1.remove_stop_sequence_at(c1)
                            route1.add_stop_sequence_at(in_index2, customer2)
                            route2.remove_stop_sequence_at(c2)
                            route2.add_stop_sequence_at(in_index1, customer1)
                            #print("exchange done")
                        neighbor.do = do
                        yield neighbor
    yield


def relocate_neighborhood(solution, customers):
    #yield
    for i in range(len(solution) - 1):
        route1 = solution[i]
        if route1.is_empty():
            continue
        for j in range(i + 1, len(solution)):
            route2 = solution[j]
            if route2.is_empty():
                continue
            for c in range(1, len(route1.stop_sequence) - 1):
                customer = route1.stop_sequence[c]
                if route2.remaining_capacity >= customer.demand:
                    for i in range(1, len(route2.stop_sequence) - 1):
                        before_customer, after_customer = route1.stop_sequence[c - 1], route1.stop_sequence[c + 1]
                        before_insertion, after_insertion = route2.stop_sequence[i - 1], route1.stop_sequence[i]
                        old_d = get_length(before_customer, customer) + get_length(customer, after_customer) + \
                                get_length(before_insertion, after_insertion)
                        new_d = get_length(before_customer, after_customer) + get_length(before_insertion, customer) + \
                                get_length(customer, after_insertion)
                        neighbor = Neighbor("relocate")
                        neighbor.diff = new_d - old_d
                        def do():
                            route1.remove_stop_sequence_at(c)
                            route2.add_stop_sequence_at(i, customer)
                            #print("relocate done")
                        neighbor.do = do
                        yield neighbor
    yield


def relocate_empty_neighborhood(solution, customers):
    for i in range(len(solution) - 1):
        route1 = solution[i]
        if route1.is_empty():
            break
        for j in range(i + 1, len(solution)):
            route2 = solution[j]
            if not route2.is_empty():
                continue
            for c in range(1, len(route1.stop_sequence) - 1):
                customer = route1.stop_sequence[c]
                before_customer, after_customer = route1.stop_sequence[c - 1], route1.stop_sequence[c + 1]
                depot = route2.stop_sequence[0]
                old_d = get_length(before_customer, customer) + get_length(customer, after_customer)
                new_d = get_length(before_customer, after_customer) + 2 * get_length(depot, customer)
                neighbor = Neighbor("relocate empty")
                neighbor.diff = new_d - old_d
                def do():
                    route1.remove_stop_sequence_at(c)
                    route2.add_stop_sequence_at(1, customer)
                    #print("relocate empty done")
                neighbor.do = do
                yield neighbor
            break
    yield


def opt_swap_2_vrp_neighborhood(solution, customers):
    #yield
    for i in range(len(solution) - 1):
        for j in range(i + 1, len(solution)):
            route1 = solution[i]
            route2 = solution[j]
            for cut1 in range(1, len(route1.stop_sequence) - 1):
                for cut2 in range(1, len(route2.stop_sequence) - 1):
                    first_slide1, last_slide1 = route1.stop_sequence[:cut1], route1.stop_sequence[cut1:]
                    first_slide2, last_slide2 = route2.stop_sequence[:cut2], route2.stop_sequence[cut2:]
                    new_load1 = sum([c.demand for c in first_slide1]) + sum([c.demand for c in last_slide2])
                    if new_load1 > route1.capacity: continue
                    new_load2 = sum([c.demand for c in first_slide2]) + sum([c.demand for c in last_slide1])
                    if new_load2 > route2.capacity: continue

                    before_cut1, after_cut1 = route1.stop_sequence[cut1 - 1], route1.stop_sequence[cut1]
                    before_cut2, after_cut2 = route2.stop_sequence[cut2 - 1], route2.stop_sequence[cut2]
                    old_d = get_length(before_cut1, after_cut1) + get_length(before_cut2, after_cut2)
                    new_d = get_length(before_cut1, after_cut2) + get_length(before_cut2, after_cut1)
                    neighbor = Neighbor("swap vrp")
                    neighbor.diff = new_d - old_d
                    def do():
                        route1.stop_sequence = list(first_slide1 + last_slide2)
                        route1.remaining_capacity = route1.capacity - new_load1
                        route2.stop_sequence = list(first_slide2 + last_slide1)
                        route2.remaining_capacity = route2.capacity - new_load2
                        #print("swap vrp done")
                    neighbor.do = do
                    yield neighbor
    yield


def insert_neighborhood(solution, customers):
    not_visited_customers = get_not_visited_customers(solution, customers)
    for customer in not_visited_customers:
        for r in solution:
            if r.remaining_capacity >= customer.demand:
                neighbor = Neighbor("insert")
                neighbor.diff = -1
                def do():
                    r.insert_stop_sequence(customer)
                    #print("insert done")
                neighbor.do = do
                yield neighbor
    yield


def local_search(solution, customers, neighborhood=opt_swap_3_neighborhood, time_limit = 600):
    tic = time.time()
    #print("Initializing local search...")
    current_solution = solution
    is_local_optimum = False
    while not is_local_optimum and time.time() - tic < time_limit:
        n = neighborhood(current_solution, customers)
        neighbor = next(n)
        while neighbor is not None:
            #print(neighbor.name)
            if neighbor.diff < -0.000001:
                neighbor.do()
                #print(neighbor.name, neighbor.diff)
                break
            neighbor = next(n)
        if neighbor is None:
            is_local_optimum = True
    #print(f"Local search Done. Time elapsed: {time.time() - tic} seconds.")
    return current_solution


def intercalate(generators, x, y):
    generators = [g(x, y) for g in generators]
    has_stopped = [False]*len(generators)
    while not all(has_stopped):
        for i, g in enumerate(generators):
            if not has_stopped[i]:
                n = next(g)
                if n is not None:
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


def GRASP(depot, customers, n_vehicles, capacity, constructor, local_search, neighborhood, time_limit=3600):
    tic = time.time()
    print("Initializing GRASP...")
    best_value = math.inf
    best_solution = []
    while time.time() - tic < time_limit:
        initial_solution = constructor(depot, customers, n_vehicles, capacity)
        #display(initial_solution)
        solution = local_search(initial_solution, customers, neighborhood, time_limit=time_limit - (time.time() - tic))
        this_value = get_solution_length(solution)
        this_value += 1000000*len(get_not_visited_customers(solution,customers))
        if this_value < best_value:
            print(f"New best solution found with value {this_value}")
            best_solution = list(solution)
            best_value = this_value
    print(F"GRASP done. Time elapsed: {time.time() - tic} seconds.")
    return best_solution


def get_trivial_solution(depot, customers):
    vehicle_tours = []

    remaining_customers = set(customers)
    remaining_customers.remove(depot)

    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand * customer_count + customer.index)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1
    return vehicle_tours


def sanitize(solution):
    for route in solution:
        route.sanitize()
    solution.sort(key=lambda r: (r.is_empty(), r.get_first().index))


def get_solution_length(solution):
    return sum([r.get_total_length() for r in solution])


def get_not_visited_customers(solution, customers):
    visited = [c for r in solution for c in r.get_customers()]
    return [c for c in customers if c not in visited]


def solve_it(input_data):
    distances.clear()
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0]

    for customer in customers:
        customer.compute_ordered_neighbors(customers)


    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = GRASP(depot, customers[1:], vehicle_count, vehicle_capacity, random_greedy, local_search,
                          lambda x,y: intercalate([insert_neighborhood, opt_swap_2_neighborhood,
                                                   opt_swap_3_neighborhood, opt_swap_2_vrp_neighborhood,
                                                   exchange_neighborhood,
                                                   relocate_empty_neighborhood], x, y), 2600)
    #sanitize(vehicle_tours)

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = 0
    for v in range(0, vehicle_count):
        route = vehicle_tours[v]
        vehicle_tour = route.stop_sequence
        if not route.is_empty():
            obj += get_length(depot,vehicle_tour[0])
            for i in range(0, len(vehicle_tour)-1):
                obj += get_length(vehicle_tour[i],vehicle_tour[i+1])
            obj += get_length(vehicle_tour[-1],depot)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += ' '.join([str(customer.index) for customer in vehicle_tours[v].stop_sequence]) + '\n'

    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

