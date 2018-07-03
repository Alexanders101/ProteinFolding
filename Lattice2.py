import numpy as np

class Lattice2():
    dim2_moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dim3_moves = [(1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]

    def __init__(self, length, dim, aa_string=np.array([]), max_size=50):
        self.dim = dim
        self.lattice = np.zeros((length, dim))
        self.current = 1
        self.prev_coord = np.array([max_size-1, max_size-1, max_size-1])
        self.directions = np.zeros(length-1)
        self.lattice[0, :self.dim] = self.prev_coord  # starts at origin
        self.coord_set = {tuple(self.prev_coord)}
        self.size = length
        if not aa_string.any():
            aa_string = np.random.choice([2, 1], self.size, True)
        self.aa_string = aa_string
        self.max_size = max_size


    def add(self, coordinates, index, type=1):  # Somewhat of a useless method unless we want to create our own protein folds.
        if np.linalg.norm(self.board[index-1, :self.dim] - coordinates) == self.dim - 1:
            self.lattice[index, :self.dim] = coordinates
            self.lattice[index, self.dim] = type  # type will be 0 for P or 1 for H
            self.current = index + 1
            self.prev_coord = coordinates
            self.coord_set.add(tuple(coordinates))
        else:
            print("Invalid placement.")

    def legal_moves(self):
        if self.dim == 2:
            next_moves = set([tuple([sum(y) for y in zip(x, self.prev_coord)]) for x in self.dim2_moves])
        else:
            next_moves = set([tuple([sum(y) for y in zip(x, self.prev_coord)]) for x in self.dim3_moves])
        return next_moves - self.coord_set

    def eval_energy(self):
        tot_energy = 0
        for i in np.arange(self.size):
            for j in np.arange(i+2, self.size):
                if self.aa_string[i]+self.aa_string[j] == 2 and np.linalg.norm(self.lattice[i, :self.dim] - self.lattice[j, :self.dim]) <= self.dim-1:
                    tot_energy -= 1
        return tot_energy

    def reset_rand(self):
        self.prev_coord = np.array([self.max_size-1, self.max_size-1, self.max_size-1])
        self.coord_set = {tuple(self.prev_coord)}
        self.lattice = np.zeros((self.size, self.dim))
        self.lattice[0, :self.dim] = self.prev_coord
        self.directions = np.zeros(self.size - 1)
        return self.make_rand()

    def make_rand(self):  # method to make a random protein fold and evaluates energy level
        for i in np.arange(1, self.size):
            self.coord_set.add(tuple(self.prev_coord))
            next_moves = self.legal_moves()
            num = len(next_moves)
            if num == 0:
                print("No where to put next amino acid.")
                return self.reset_rand()
            curr = np.array(list(next_moves)[int(np.random.choice(np.arange(num), size=1))])
            if max(curr) > self.max_size*2-2 or min(curr) < 0:
                print("Went off the grid.")
                return self.reset_rand()
            move = tuple(curr - self.prev_coord)
            for index, elem in enumerate(self.dim3_moves):
                if move == elem:
                    self.directions[i-1] = index
            self.lattice[i, :self.dim] = curr
            self.prev_coord = curr
        self.current = self.size
        return self.eval_energy()

    def printer(self):  # This was useful in debugging.
        if self.dim == 3:
            print("3D lattice would be hard to print")
        if self.dim == 2:
            mat = [["x" for _ in np.arange(self.size*2)] for _ in np.arange(self.size*2)]
            for x in np.arange(self.size):
                mat[self.size-int(self.lattice[x, 1])][self.size+int(self.lattice[x, 0])] = str(int(self.aa_string[x]))
            for x in mat:
                print(x)