from random import random, randint, sample
from math import floor, log, log2
import numpy as np
from copy import deepcopy
import time



class Node():
    def __init__(self, function, con1, con2):
        self.function = function
        self.con1 = con1
        self.con2 = con2

    def compute(self, arg1, arg2):
        if self.function == 0:
            return arg1 + arg2
        elif self.function == 1:
            return arg1 - arg2
        elif self.function == 2:
            return arg1 * arg2
        elif self.function == 3:
            return arg1 / arg2 if arg2 != 0 else arg1
        else:
            raise Exception("Invalid code for mathematical operation")

    @staticmethod
    def random_node(max_arg):
        return Node(randint(0,3), randint(0, max_arg), randint(0, max_arg))

    def mutate(self, element, max_arg):
        if element == 0:
            self.function = randint(0, 3)
        elif element == 1:
            self.arg1 = randint(0, max_arg)
        elif element == 2:
            self.arg2 = randint(0, max_arg)
        else:
            raise Exception("Invalid element to mutate")

    def to_string(self):
        return str(self.function) + "-" + str(self.con1) + "-" + str(self.con2)










class Individual():
    def __init__(self, num_inputs, num_rows, num_cols, num_outputs):
        self.num_inputs = num_inputs
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_outputs = num_outputs
        self.num_nodes = num_rows * num_cols
        self.nodes = self._generate_random_nodes()
        self.outputs = self._generate_random_outputs()
        self.node_outputs = [-1 for _ in range(self.num_inputs+self.num_nodes)]
        self.to_evaluate = [False for _ in range(self.num_inputs+self.num_nodes)]
        self._identify_to_evaluate()

    def _generate_random_nodes(self):
        nodes = []
        for i in range(self.num_nodes):
            max_arg = self.num_inputs + int(i / self.num_rows) * self.num_cols - 1
            nodes.append(Node.random_node(max_arg))
        return nodes

    def _generate_random_outputs(self):
        outputs = [randint(0, self.num_inputs+self.num_nodes-1) for _ in range(self.num_outputs)]
        return outputs

    def mutate(self, num_mutations):
        for i in range(num_mutations):
            rand = randint(0, 3 * self.num_nodes + self.num_outputs - 1)
            if rand < 3 * self.num_nodes:
                node = int(rand / 3)
                element = rand % 3
                max_arg = self.num_inputs + int(node / self.num_rows) * self.num_cols - 1
                self.nodes[node].mutate(element, max_arg)
            else:
                idx = rand - 3 * self.num_nodes
                self.outputs[idx] = randint(0, self.num_nodes + self.num_inputs - 1)
        self._identify_to_evaluate()

    def _identify_to_evaluate(self):
        self.to_evaluate = [False for _ in range(self.num_inputs+self.num_nodes)]
        for i in range(self.num_outputs):
            self.to_evaluate[self.outputs[i]] = True

        for i, node in reversed(list(enumerate(self.nodes))):
            if self.to_evaluate[i]:
                x = node.con1
                y = node.con2
                self.to_evaluate[x] = True
                self.to_evaluate[y] = True

    def _load_inputs(self):
        for i in range(self.num_inputs):
            self.node_outputs[i] = self.inputs[i]

    def _execute(self):
        for i in range(self.num_nodes):
            if self.to_evaluate[self.num_inputs+i]:
                arg1 = self.node_outputs[self.nodes[i].con1]
                arg2 = self.node_outputs[self.nodes[i].con2]
                node_output = self.nodes[i].compute(arg1, arg2)
                self.node_outputs[self.num_inputs+i] = node_output

    def _get_outputs(self):
        outputs = []
        for i in range(self.num_outputs):
            outputs.append(self.node_outputs[self.outputs[i]])
        return outputs

    def get_outputs_from_inputs(self, inputs):
        self.inputs = inputs
        self._load_inputs()
        self._execute()
        outputs = self._get_outputs()
        return outputs

    @staticmethod
    def copy(individual):
        copy = deepcopy(individual)
        nodes = []
        for i in range(len(individual.nodes)):
            node = deepcopy(individual.nodes[i])
            nodes.append(node)
        copy.nodes = nodes
        return copy

    def to_string(self):
        ret = ""
        for i in range(0, len(self.nodes)):
            ret += self.nodes[i].to_string() + " "
        ret += (",").join(str(i) for i in self.outputs)
        return ret










class Population():
    def __init__(self, size, num_inputs=24, num_rows=5, num_cols=5, num_outputs=12, num_mutations=3, calculate_fitness=()):
        self.size = size
        self.num_mutations = num_mutations
        self.calculate_fitness = calculate_fitness
        self.individuals = [Individual(num_inputs, num_rows, num_cols, num_outputs) for _ in range(size)]
        self.fitnesses = [10 ** 5 for _ in range(size)]

    def evolve(self, criteria, num_iterations):
        print("Started evolve...")
        last_best_fitness = 10000
        for i in range(num_iterations):
            self._calculate_fitness()
            best_fitness, best_fitness_idx = self._get_best_individual()
            if best_fitness_idx > 0:
                self.individuals[0] = self.individuals[best_fitness_idx]
            if best_fitness < criteria:
                break
            if best_fitness != last_best_fitness or i % 100 == 0:
                print("Generation #" + str(i) + ": best fitness: " + str(best_fitness))
                print("*"*40)
            last_best_fitness = best_fitness
            for i in range(1, self.size):
                copy = Individual.copy(self.individuals[i])
                copy.mutate(self.num_mutations)
                self.individuals[i] = copy
        return self.individuals[best_fitness_idx]

    def _calculate_fitness(self):
        for i in range(self.size):
            self.fitnesses[i] = self.calculate_fitness(self.individuals[i])
            #print("Individual #"+str(i)+": fitness: "+str(self.fitnesses[i]))

    def _get_best_individual(self):
        best_fitness = min(self.fitnesses)
        best_individual_idx = max(index for index, item in enumerate(self.fitnesses) if item == best_fitness)
        return best_fitness, best_individual_idx










class Game():
    def __init__(self, tiers=0, stacks=0, verbose=False, file_path=''):
        self.verbose = verbose
        self.in_progress = True
        self.curr_lowest = 1
        self.move = 0

        if file_path != '':
            with open(file_path) as f:
                lines = f.readlines()
                lines = [list(map(int, line.split())) for line in lines]
                self.tiers = len(lines)
                self.stacks = len(lines[0])
                self.state = np.reshape(lines, (self.tiers, self.stacks))
                self.check_remove()
                if self.verbose: self.print_state()
        else:
            self.tiers = tiers
            self.stacks = stacks
            self._randomize_game()

    def _randomize_game(self):
        contrainers_num = self.tiers*self.stacks
        initial_state = sample(range(1, contrainers_num+1), contrainers_num)
        initial_state = np.reshape(initial_state, (self.tiers, self.stacks))
        empty_rows = np.zeros((2, self.stacks), dtype=int)
        self.state = np.vstack([empty_rows, initial_state])
        self.check_remove()
        if self.verbose: self.print_state()

    def check_remove(self):
        i = 0
        while i < self.stacks:
            top = self._get_top_container_index(i)
            if top is not None and self.state[top][i] == self.curr_lowest:
                self.state[top][i] = 0
                if self.verbose: self.print_state()
                self.curr_lowest += 1
                i = 0
            else:
                i += 1
        for j in range(0, self.stacks):
            if self._get_top_container_index(j) is not None:
                return
        self.in_progress = False
        if self.verbose:
            print("Congratulation, you won!")
            print("Number of moves: ", self.move)

    def make_move(self, from_stack, to_stack):
        from_top = self._get_top_container_index(from_stack)
        to_top = self._get_top_container_index(to_stack)
        if from_top is None or to_top == 0:
            if self.verbose:
                print("Illegal move: from_stack is empty or to_stack is full!")
            else:
                raise Exception("Illegal move: from_stack is empty or to_stack is full!")
            return
        if to_top is None:
            to_top = self.stacks + 2

        container_num = self.state[from_top][from_stack]
        self.state[from_top][from_stack] = 0
        self.state[max(0,to_top-1)][to_stack] = container_num
        self.move += 1

        if self.verbose: self.print_state()
        self.check_remove()

    def print_state(self):
        print("========== New state! Move: "+str(self.move)+" ==========")
        print(self.state)
        print("========================================")

    def _get_top_container_index(self, stack):
        col = self.state[:,stack]
        return next((i for i, x in enumerate(col) if x), None)

    def is_solved(self):
        return not self.in_progress










def brp_fitness(individual: Individual, games, mins):
    fitness = 0
    for i in range(len(games)):
        penalty = 0
        while not games[i].is_solved() and games[i].move + penalty < 50:
            outputs = individual.get_outputs_from_inputs(list(games[i].state.flatten()))
            min_output = min(outputs)
            min_index = outputs.index(min_output)
            division = games[i].stacks - 1
            a = floor(min_index/division)
            b = min_index % division
            try:
                games[i].make_move(a, b if a > b else b+1)
            except:
                penalty += 5
        fitness += log2(games[i].move + penalty) / log2(mins[i])
    return fitness

def calculate_moves(individual):
    games = [Game(file_path='./games/3x3/game1.txt'), Game(file_path='./games/3x3/game2.txt'),
             Game(file_path='./games/3x3/game3.txt'), Game(file_path='./games/3x3/game4.txt'),
             Game(file_path='./games/3x3/game5.txt'), Game(file_path='./games/3x3/game6.txt'),
             Game(file_path='./games/3x3/game7.txt'), Game(file_path='./games/3x3/game8.txt'),
             Game(file_path='./games/3x3/game9.txt'), Game(file_path='./games/3x3/game10.txt')]
    for i in range(len(games)):
        penalty = 0
        while not games[i].is_solved() and games[i].move + penalty < 40:
            outputs = individual.get_outputs_from_inputs(list(games[i].state.flatten()))
            min_output = min(outputs)
            min_index = outputs.index(min_output)
            division = games[i].stacks - 1
            a = floor(min_index / division)
            b = min_index % division
            try:
                games[i].make_move(a, b if a > b else b + 1)
            except:
                penalty += 5
        print("Game #", i, ": ", games[i].move + penalty)

def train_cgp():
    def calculate_fitness(individual):
        games = [Game(file_path='./games/3x3/game1.txt'), Game(file_path='./games/3x3/game2.txt'),
                 Game(file_path='./games/3x3/game3.txt'), Game(file_path='./games/3x3/game4.txt'),
                 Game(file_path='./games/3x3/game5.txt'), Game(file_path='./games/3x3/game6.txt'),
                 Game(file_path='./games/3x3/game7.txt')]
        mins = [6, 5, 3, 4, 1.01, 6, 6]
        return brp_fitness(individual, games, mins)

    pop = Population(size=10, num_inputs=15, num_rows=20, num_cols=10, num_outputs=6, num_mutations=30, calculate_fitness=calculate_fitness)
    best_individual = pop.evolve(1, 10000)
    calculate_moves(best_individual)
    print(best_individual.to_string())





def play_game():
    print("*"*10 + "BLOCK RELOCATION PUZZLE" + "*"*10)
    print("@author: Josip Ivančević")
    print("@menthor: Domagoj Jakobović")
    print("FER, 2022.")
    input("Press Enter to continue...")
    print()
    print("Please choose number of tiers and stacks")
    tiers = int(input("Number of tiers:"))
    stacks = int(input("Number of stacks:"))
    game = Game(tiers, stacks, verbose=True, file_path='./games/3x3/game1.txt')
    while (game.in_progress):
        from_stack = int(input("Choose from_stack:"))
        to_stack = int(input("Choose to_stack:"))
        game.make_move(from_stack, to_stack)





def main():
    correct_input = False
    while not correct_input:
        command = input("Press 1 to train cgp, or 2 to play game")
        if int(command) == 1:
            correct_input = True
            train_cgp()
        elif int(command) == 2:
            correct_input = True
            play_game()
        else:
            continue

if __name__ == '__main__':
    main()

