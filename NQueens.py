import numpy as np
from math import exp


class bcolors:
    """ Class used to print colored text in the terminal"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class NQueens:
    def __init__(self, beta=1, N=8, q_coordinates = [], q_indices = [], chosen = 0) -> None:
        """ Represent an NxN board with N queens.
        @param beta: parameter used during the algorithm
        @param N: number of rows and columns of the chessboard
        @param coords: a list of each queens 2D positions on the board
        @param indices: a list of each queens 1D positions on the board
        @param id_chosen: id of the queen that will be moved
        """
        self.beta = beta
        self.N = N
        self.q_coordinates = q_coordinates
        self.q_indices = q_indices
        self.chosen = chosen
        # the number of conflicts on the board correspond to the sum of the number of conflicts for each queen
        self.num_conflicts = 0
    

    # This property must be used to place queens on the board
    def is_safe(self) -> bool:
        """ This function asserts that there exists no row or column with more than one queen on it."""
        for i in range(self.N):
            if sum(self.q_coordinates[:, 0] == i) > 1 or sum(self.q_coordinates[:, 1] == i) > 1:
                return False
        return True
    

    # default initialisation (most likely not optimal)
    def main_diagonal_initialisation(self) -> tuple:
        """ This function initialises the board with all the queens on the main diagonal. """
        self.q_coordinates = np.array([(i,i) for i in range(self.N)])
        self.q_indices = np.array([i*(self.N + 1) for i in range(self.N)])
        self.num_conflicts = self.diagonal_conflict_calculator()
        assert self.is_safe()
        return (self.q_coordinates, self.q_indices)
    

    def default_se_knight_movement(self, x, y) -> tuple:
        """ This function returns a fixed new position from an initial starting position with respect to the south-east knight movement. """
        if self.N % 2 == 0 and y == self.N - 2:
            return (x+1, 1)
        return (x+1, (y+2) % self.N)
    
    
    # TODO: work on placing queens at distinct positions
    def random_knight_movement(self, x, y) -> tuple:
        """ This function returns a randomly chosen new position from an initial starting position with respect to knight movements. """
        sign = np.random.choice([-1, 1])
        weight = np.random.choice([1, 2])
        return ((x + sign*weight) % self.N, (y + sign*(3-weight)) % self.N)

    
    # better initialisation based on knight movements
    # right now, the initialisation is fixed
    def knight_initialisation(self) -> tuple:
        """ This function initialises the board with queens positioned using knight movements."""
        # initialise the first queen's position at (0,0)
        self.q_coordinates = np.array([(0,0)])
        self.q_indices = np.array([0])
        # place the remaining queens using knight movements
        x, y = 0, 0
        for i in range(1, self.N):
            x, y = self.default_se_knight_movement(x, y)
            self.q_coordinates = np.append(self.q_coordinates, [(x, y)], axis=0)
            self.q_indices = np.append(self.q_indices, [x + i*self.N])
        self.num_conflicts = self.diagonal_conflict_calculator()
        assert self.is_safe()
        return (self.q_coordinates, self.q_indices)
    

    def single_queen_conflict_calculator(self, queen_id) -> int:
        """ This function calculates the number of conflicts for a single queen."""
        conflicts = 0
        for i in range(self.N):
            if i != queen_id:
                if abs(self.q_coordinates[queen_id][0]-self.q_coordinates[i][0]) == abs(self.q_coordinates[queen_id][1]-self.q_coordinates[i][1]):
                    conflicts += 1
        return conflicts


    def diagonal_conflict_calculator(self) -> int:
        """ This function calculates the total number of conflicts on the board."""
        return sum([self.single_queen_conflict_calculator(_) for _ in range(self.N)])
    

    def display_board(self) -> None:
        """ This function displays the board."""
        # print the board
        # 0 = empty square, 1 = queen
        # if the queen is conflicting with another, print it red
        board = np.zeros((self.N, self.N))
        for i in range(self.N):
            board[self.q_coordinates[i][0], self.q_coordinates[i][1]] = 1
        qid = 0
        for i in range(self.N):
            for j in range(self.N):
                if board[i, j] == 1:
                    # check for conflict
                    if self.single_queen_conflict_calculator(qid) > 0:
                        print(bcolors.FAIL + '1' + bcolors.ENDC, end=' ')
                    else:
                        print(bcolors.OKGREEN + '1' + bcolors.ENDC, end=' ')
                    qid += 1
                else:
                    print("0", end=" ")
            print()
        return

    def move(self):
        # 1) pick uniformely at random a couple of queens
        # 2) calculate the acceptance probability by using the conflict function
        # 3) check if the move has to be done

        flag = True
        while flag == True:
            id_1 = np.random.randint(0,self.N)
            id_2 = np.random.randint(0,self.N)
            if id_1 != id_2:
                flag = False


        r1_old, c1_old = self.q_coordinates[id_1][0], self.q_coordinates[id_1][1]
        r2_old, c2_old = self.q_coordinates[id_2][0], self.q_coordinates[id_2][1]

        eventual_conflict = self.num_conflicts - 2*self.single_queen_conflict_calculator(id_1) \
        -2*self.single_queen_conflict_calculator(id_2)

        r1_new, c1_new = r2_old, c1_old
        r2_new, c2_new = r1_old, c2_old

        self.q_coordinates[id_1][0], self.q_coordinates[id_1][1] = r1_new, c1_new
        self.q_coordinates[id_2][0], self.q_coordinates[id_2][1] = r2_new, c2_new

        eventual_conflict += 2*self.single_queen_conflict_calculator(id_1) \
        +2*self.single_queen_conflict_calculator(id_2)

        a = min(1, exp(-self.beta*(eventual_conflict-self.num_conflicts))) # acceptance probability

        if (np.random.uniform() < a):
            # in this case the move is made
            self.num_conflicts = eventual_conflict
        else:
            # the move is not made
            self.q_coordinates[id_1][0], self.q_coordinates[id_1][1] = r1_old, c1_old
            self.q_coordinates[id_2][0], self.q_coordinates[id_2][1] = r2_old, c2_old



if __name__ == "__main__":
    board = NQueens(beta = 1, N=50)
    board.knight_initialisation()
    print(f'---Board initialisation of size: {board.N}x{board.N}---')
    print(f'- 2D positions of each queen: {board.q_coordinates}')
    print(f'- 1D positions of each queen: {board.q_indices}')
    print(f'- Total conflicts: {board.num_conflicts}')
    print('- Board:')
    #board.display_board()

    while(board.num_conflicts != 0):
        board.move()
        print(board.num_conflicts)
    assert board.is_safe()
    #print(board.q_coordinates)

