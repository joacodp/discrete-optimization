#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import time
import networkx as nx
import matplotlib.pyplot as plt
import sys


# Defining a Class
class GraphVisualization:

    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.show()



def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    sys.setrecursionlimit(1100)
    #G = GraphVisualization()
    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
    print(f"{len(edges)} edges found")
    nodes = [Node(i) for i in range(node_count)]
    for edge in edges:
        node1, node2 = nodes[edge[0]], nodes[edge[1]]
        node1.add_edge(node2)
        #G.addEdge(edge[0], edge[1])

    #G.visualize()
    graph = Graph(nodes)

    # Find max clique
    clique = find_single_clique(graph)

    # Build heuristic solution
    first_solution, upper_bound = heuristic_painting(graph)

    # Search with constraint programming
    solution, number_of_colors, is_optimal = search(graph, clique, first_solution, upper_bound, 2*3600)

    # build a trivial solution
    # every node has its own color
    #solution = range(0, node_count)

    # prepare the solution in the specified output format
    output_data = str(number_of_colors) + ' ' + str(int(is_optimal)) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def search(graph, max_clique, best_solution, upper_bound, time_limit):
    tic = time.time()
    fail = False
    lower_bound = len(max_clique)
    print(f"Starting search with upper bound {upper_bound} and lower bound {lower_bound}.")

    is_feasible = True
    try:
        while upper_bound > lower_bound and is_feasible:
            graph.reset(upper_bound - 1)
            paint_clique(max_clique)
            cs = ConstraintStore()

            first_node = graph.get_most_constrained_node()
            is_feasible = recursive_search(graph, first_node, cs, tic, time_limit)
            if is_feasible:
                print(f"New solution found with {upper_bound - 1} colors")
                upper_bound -= 1
                best_solution = graph.get_solution()
    except Exception as e:
        print(f"There was an error: {e}")
        fail = True


    tac = time.time()
    return best_solution, upper_bound, tac - tic <= time_limit and not fail


def recursive_search(graph, node, cs, tic, time_limit):
    colors = node.possible_colors.copy()
    for color in colors:
        if time.time() - tic > time_limit:
            break
        is_feasible, constraint = cs.try_add_equality_constraint(node, color, True)
        if is_feasible:
            if graph.all_painted():
                return True
            else:
                new_node = graph.get_most_constrained_node()
                is_feasible = recursive_search(graph, new_node, cs, tic, time_limit)
                if is_feasible:
                    return True
                else:
                    cs.backtrack(constraint)
        else:
            cs.backtrack(constraint)
    return False


def find_single_clique(graph):
    clique = []
    vertices = list(graph.nodes)
    rand = random.randrange(0, len(vertices), 1)
    clique.append(vertices[rand])
    for v in vertices:
        if v in clique:
            continue
        isNext = True
        for u in clique:
            if u in v.neighbours:
                continue
            else:
                isNext = False
                break
        if isNext:
            clique.append(v)

    return clique


def heuristic_painting(graph):
    upper_bound = 0
    graph.nodes.sort(key=lambda x: len(x.neighbours), reverse=True)
    for node in graph.nodes:
        color = 0
        painted = False
        neighbours_colors = node.get_neighbours_colors()
        while not painted:
            if color not in neighbours_colors:
                painted = True
                node.color = color
                if color > upper_bound:
                    upper_bound = color
            else:
                color += 1

    return graph.get_solution(), upper_bound + 1


def paint_clique(clique):
    i = 0
    for node in clique:
        node.color = i
        for node in node.neighbours:
            node.possible_colors.remove(i)
        i += 1


class Node:
    def __init__(self, n):
        self.index = n
        self.color = None
        self.possible_colors = set()
        self.amount_of_colors = 0
        self.neighbours = set()

    def add_edge(self, neighbour):
        self.neighbours.add(neighbour)
        neighbour.neighbours.add(self)

    def get_unpainted_neighbours(self):
        return [n for n in self.neighbours if n.color is None]

    def get_neighbours_colors(self):
        return set([n.color for n in self.neighbours if n.color is not None])

    def reset(self, amount_of_colors):
        if amount_of_colors is not None:
            self.possible_colors = set([i for i in range(amount_of_colors)])
            self.amount_of_colors = amount_of_colors
        self.color = None


class Graph:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_unpainted_nodes(self):
        return [n for n in self.nodes if n.color is None]

    def get_most_constrained_node(self):
        return sorted(self.get_unpainted_nodes(), key= lambda x: (len(x.possible_colors), -len(x.neighbours)))[0]

    def all_painted(self):
        return all([n.color is not None for n in self.nodes])

    def is_feasible(self):
        for node in self.nodes:
            if node.color in node.get_neighbours_colors():
                return False
        return True

    def dfs_util(self, temp, v, visited):

        # Mark the current vertex as visited
        visited[v.index] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in v.neighbours:
            if visited[i.index] == False:
                # Update the list
                temp = self.dfs_util(temp, i, visited)
        return temp

    # Method to retrieve connected components
    # in an undirected graph
    def connected_components(self):
        visited = []
        cc = []
        for i in range(len(self.nodes)):
            visited.append(False)
        for v in self.nodes:
            if visited[v.index] == False:
                temp = []
                cc.append(self.dfs_util(temp, v, visited))
        return cc

    def get_solution(self):
        self.nodes.sort(key= lambda x: x.index)
        return [n.color for n in self.nodes]

    def reset(self, amount_of_colors= None):
        for node in self.nodes:
            node.reset(amount_of_colors)


class Constraint:
    def __init__(self, type, node, color, is_choice):
        self.type = type
        self.node = node
        self.color = color
        self.is_choice = is_choice

    def __eq__(self, other):
        return self.node.index == other.node.index and self.color == other.color and self.type == other.type

    def __hash__(self):
        return hash((self.type, self.node.index, self.color))


class ConstraintStore:
    def __init__(self):
        self.constraints_stack = []
        self.constraints_hashset = set()

    def try_add_equality_constraint(self, node, color, is_choice):
        constraint = Constraint("equality", node, color, is_choice)
        self.constraints_stack.append(constraint)
        self.constraints_hashset.add(constraint)
        if color not in node.possible_colors or node.color is not None:
            return False, constraint
        else:
            node.color = color
            for neighbour in node.neighbours:
                ret = self.try_add_inequality_constraint(neighbour, color, False)
                if not ret:
                    return False, constraint
            return True, constraint

    def try_add_inequality_constraint(self, node, color, is_choice):
        if node.color == color:
            return False

        constraint = Constraint("inequality", node, color, is_choice)
        if constraint not in self.constraints_hashset:
            self.constraints_stack.append(constraint)
            self.constraints_hashset.add(constraint)
            if color in node.possible_colors:
                node.possible_colors.remove(color)
        ret = len(node.possible_colors) > 0
        return ret

    def debug_node_with_color(self, node, color):
        return any([ne for ne in node.neighbours if ne.color == color])

    def debug_node_without_color(self, node):
        print(sorted(node.get_neighbours_colors()))
        return all([any([ne for ne in node.neighbours if ne.color == color]) for color in range(node.amount_of_colors)])

    def backtrack(self, c):
        #is_choice = False
        this_constraint = None
        #while not is_choice:
        while this_constraint is None or this_constraint != c:
            this_constraint = self.constraints_stack.pop(-1)
            self.constraints_hashset.remove(this_constraint)
            #is_choice = constraint.is_choice
            if this_constraint.type == "equality":
                this_constraint.node.color = None
            elif this_constraint.type == "inequality":
                this_constraint.node.possible_colors.add(this_constraint.color)
            else:
                print("not valid constraint type")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')






















    """def add_color(self, node, color):
        if color in node.prohibited_colors:
            return False
        else:
            is_feasible = True
            changed_nodes = []
            for neighbour in node.get_unpainted_nodes():
                if color not in node.prohibited_colors:
                    is_feasible = self.remove_color(neighbour, color)
                    changed_nodes.append(neighbour)
                    if not is_feasible:
                        break

            if not is_feasible:
                for changed_node in changed_nodes:
                    changed_node.prohibited_colors.remove(color)
                return False
            else:
                node.color = color
                return True

    def remove_color(self, node, color):
        node.prohibited_colors.add(color)
        if len(self.available_colors) == len(node.prohibited_colors):
            return False
        else:
            return True"""