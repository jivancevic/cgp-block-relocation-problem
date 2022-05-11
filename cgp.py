from random import random, randint, sample
#from math import floor, prod
import numpy as np

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
            return arg1 / arg2
        else:
            raise Exception("Invalid code for mathematical operation")

    def mutate(self, element, max_arg):
        if element == 0:
            self.function = randint(0, 3)
        elif element == 1:
            self.arg1 = randint(0, max_arg)
        elif element == 2:
            self.arg2 = randint(0, max_arg)
        else:
            raise Exception("Invalid element to mutate")


class Genotype():
    def __init__(self, num_inputs, num_rows, num_cols, num_outputs):
        self.num_inputs = num_inputs
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_outputs = num_outputs
        self.num_nodes = num_rows * num_cols
        self.inputs = []
        self.nodes = self._generate_random_nodes()
        self.outputs = [-1 for _ in range(self.num_outputs)]
        self.node_outputs = [-1 for _ in range(self.num_inputs+self.num_nodes)]
        self.to_evaluate = [False for i in self.num_nodes]

    def _generate_random_nodes(self):
        for _ in range(self.num_nodes):
            i = True


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
                self.outputs[idx] = randint(0, self.num_inputs + 3 * self.num_nodes - 1)
        self.decode()

    def _identify_to_evaluate(self):
        for i in range(self.num_outputs):
            self.to_evaluate[self.outputs[i]] = True

        for i, node in reversed(list(enumerate(self.nodes))):
            if self.to_evaluate[i]:
                x = node.arg1
                y = node.arg2
                self.to_evaluate[x] = True
                self.to_evaluate[y] = True

    def _load_inputs(self):
        for i in range(self.num_nodes):
            self.node_outputs.append(self.inputs[i])

    def _execute(self):
        for i in range(self.num_nodes):
            if self.to_evaluate[self.num_inputs+i]:
                arg1 = self.node_outputs[self.nodes[i].con1]
                arg2 = self.node_outputs[self.nodes[i].con2]
                node_output = self.nodes[i].compute(arg1, arg2)
                self.node_outputs[self.num_inputs+i] = node_output

    def decode(self):
        self.node_outputs = []
        self._identify_to_evaluate()
        self._load_inputs()
        self._execute()

    def calculate_fitness(self, games):
        for game in games:
            while not game.is_solved():
                self.inputs = self._decode_inputs(game)




    def _decode_inputs(self, game):
        return list(game.state.flatten())

    def _decode_outputs(self):
        pass


class Population():
    def __init__(self, games, size, num_inputs=24, num_rows=5, num_cols=5, num_outputs=12):
        self.games = games
        self.genotypes = [Genotype(num_inputs, num_rows, num_cols, num_outputs) for _ in size]
        self.fitnesses = [10 ** 5 for _ in size]

    def evolve(self, criteria, num_iterations):
        for i in range(num_iterations):
            self._calculate_fitness()
            best_fitness, best_fitness_idx = self._get_best_fitness()
            if best_fitness < criteria:
                break

        return self.genotypes[best_fitness_idx]

    def _calculate_fitness(self):
        for i in range(self.size):
            self.fitnesses[i] = self.genotypes[i].calculate_fitness(self.games)


    def _get_best_genotype(self):
        best_genotype = min(self.genotypes, key=self.genotypes.get)
        return best_genotype, self.genotypes[best_genotype]


class Game():
    def __init__(self, tiers, stacks, verbose=False):
        self.tiers = tiers
        self.stacks = stacks
        self.verbose = verbose
        self.in_progress = True
        self.curr_lowest = 1
        self.move = 0
        self._initialize_game()

    def _initialize_game(self):
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
            print(i, top)
            if top is not None and self.state[top][i] == self.curr_lowest:
                self.state[top][i] = 0
                if self.verbose: self.print_state()
                self.curr_lowest += 1
                i = 0
            else:
                i += 1

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
    game = Game(tiers, stacks)
    while (game.in_progress):
        from_stack = int(input("Choose from_stack:"))
        to_stack = int(input("Choose to_stack:"))
        game.make_move(from_stack, to_stack)


if __name__ == '__main__':
    play_game()

