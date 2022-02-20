#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])
from pulp import *


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])
    print(capacity)
    optimal = False
    value = 0

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    taken_dict = [0] * len(items)
    items.sort(key=lambda x: x.value/x.weight, reverse=True)

    value, taken = heuristic_density(items, capacity)
    optimal, value, taken = solve_bb(items, capacity, "DFS", 3600*4.5, taken, value)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(int(optimal)) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def heuristic_density(items, capacity):
    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0] * len(items)

    items.sort(key=lambda x: x.value/x.weight, reverse=True)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return value, taken


def heuristic_density_continuous(items, capacity):
    # a heuristic (still exact) algorithm to solve the continuous knapsack
    value = 0
    weight = 0
    taken = []

    #items.sort(key=lambda x: x.value/x.weight, reverse=True)

    for item in items:
        if weight + item.weight <= capacity:
            taken.append(item)
            value += item.value
            weight += item.weight
            if weight == capacity:
                return True, value, taken, None
        else:
            var_value = (capacity - weight)/item.weight
            value += item.value*var_value
            return False, value, taken, item
    return True, value, taken, None


def solve_bb(items, capacity, type, time_limit, initial_solution, initial_value):
    best_integer_value = initial_value
    print(f"Integer solution found: {best_integer_value}")
    best_integer_solution = [i for i in items if initial_solution[i.index] == 1]
    end_criteria = False
    taken = [0] * len(items)
    nodes_visited = 0

    queue = [Node(items, capacity, {}, sum([item.value for item in items]), "0")]
    tic = time()
    while not end_criteria:
        if nodes_visited%100000 == 0:
            print(f"Nodes visited: {nodes_visited}")
        if type == "DFS":
            this_node = queue.pop()
        elif type == "BFS":
            this_node = queue.pop(0)
        else:
            raise Exception("bad b&b type")
        #print(this_node.name)

        is_integral = this_node.branch(best_integer_value, choose_less_integral)
        nodes_visited += 1
        if is_integral and this_node.objective_value > best_integer_value:
            best_integer_value = this_node.objective_value
            best_integer_solution = this_node.solution
            print(f"Integer solution found: {best_integer_value}, weight: {sum([i.weight for i in best_integer_solution])} ")
        else:
            queue += this_node.children
        seconds_elapsed = time() - tic
        end_criteria = len(queue) == 0 or seconds_elapsed > time_limit

    for i in best_integer_solution:
        taken[i.index] = 1
    seconds_elapsed = time() - tic
    print(f"Elapsed time: {seconds_elapsed} seconds.")
    if len(queue) == 0:
        return True, best_integer_value, taken
    else:
        return False, best_integer_value, taken


def solve_linear(items, capacity):
    # Returns status and a dict with key=item, value=variableValue at optimum

    p = LpProblem("P", LpMaximize)
    variable_dict = LpVariable.dicts("v", items, 0, 1)
    p += lpSum([item.value*variable_dict[item] for item in items])
    p += lpSum([item.weight*variable_dict[item] for item in items]) <= capacity
    status = p.solve(PULP_CBC_CMD(msg=False))

    return status, p.objective.value(), {item: v.varValue for item, v in variable_dict.items()}


def choose_less_integral(values_by_item):
    non_integral = {item: abs(value - round(value)) for item, value in values_by_item.items()
                    if not float(value).is_integer()}
    if len(non_integral) == 0:
        return True, None
    else:
        item = max(non_integral, key=non_integral.get)
        return False, item


class Node:

    def __init__(self, items, capacity, fixed_variables, upper_bound, name):
        self.items = items
        self.capacity = capacity
        self.fixed_variables = fixed_variables
        self.upper_bound = upper_bound
        self.name = name
        self.objective_value = 0
        self.solution = [i for i, value in fixed_variables.items() if value == 1]
        self.children = []

    def branch(self, best_integer, chooser):
        # Returns is_integral (true), objective_value (float), children_nodes (list of nodes)

        # chooser is a function that takes dictionary item: value and selects one item and returns it

        if self.upper_bound <= best_integer:
            return False

        free_items = [i for i in self.items if i not in self.fixed_variables.keys()]
        if len(free_items) == 0:
            self.objective_value = sum([item.value for item, v in self.fixed_variables.items() if v == 1])
            return True

        # status, objective_value, variables = solve_linear(free_items, self.capacity)
        all_integral, objective_value, taken, cut_item = heuristic_density_continuous(free_items, self.capacity)
        real_objective_value = objective_value + sum([item.value for item, v in self.fixed_variables.items() if v == 1])
        self.objective_value = real_objective_value

        """if status != LpStatusOptimal:
            return False"""

        if real_objective_value <= best_integer:
            return False

        # all_integral, item = chooser(variables)
        if all_integral:
            self.solution += taken
            return True

        if self.capacity >= cut_item.weight:
            fixed_variables1 = self.fixed_variables.copy()
            fixed_variables1[cut_item] = 1
            self.children.append(Node(self.items, self.capacity - cut_item.weight, fixed_variables1,
                                      real_objective_value, self.name + "1"))

        fixed_variables0 = self.fixed_variables.copy()
        fixed_variables0[cut_item] = 0
        self.children.append(Node(self.items, self.capacity, fixed_variables0,
                                  real_objective_value, self.name + "0"))

        self.objective_value = real_objective_value

        return False


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

