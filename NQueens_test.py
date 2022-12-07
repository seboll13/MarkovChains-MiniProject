import numpy as np
from math import exp
from time import time

'''
@todo
1) optimise the swapping method -> our curr bottle-neck
'''


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
        # dictionnaries of (queen_id, was_conflicting) pairs
        self.conflicting_queens = {}
        self.non_conflicting_queens = {}
    

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
        self.num_conflicts = self.N * (self.N - 1)
        #self.conflicting_queens = {_: self.N-1 for _ in range(self.N)}
        for i in range(self.N):
            self.conflicting_queens.update({i:self.N-1})
        assert self.is_safe()
        return (self.q_coordinates, self.q_indices)
    

    def random_positions_initialisation(self) -> tuple:
        """ This function randomly puts queens on the board, whilst preserving the no row/column conflict contraint. """
        a, b = np.arange(self.N), np.arange(self.N)
        np.random.shuffle(a)
        np.random.shuffle(b)
        self.q_coordinates = np.array([(a[i], b[i]) for i in range(self.N)])
        self.q_indices = np.array([a[i] + i*self.N for i in range(self.N)])
        self.num_conflicts = self.diagonal_conflict_calculator()
        assert self.is_safe()
        return (self.q_coordinates, self.q_indices)
    

    def default_se_knight_movement(self, x, y) -> tuple:
        """ This function returns a fixed new position from an initial starting position with respect to the south-east knight movement. """
        if self.N % 2 == 0 and y == self.N - 2:
            return (x+1, 1)
        return (x+1, (y+2) % self.N)

    
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
    

    def swap_queens(self) -> None:
        """ This function selects two queens at random from the board and swaps their vertical positions. """
        # select two queens at random
        q1_id, q2_id = np.random.choice(self.N, 2, replace=False)
        # swap the y coordinates of both queens
        self.q_coordinates[q1_id][1], self.q_coordinates[q2_id][1] = self.q_coordinates[q2_id][1], self.q_coordinates[q1_id][1]
        # update the indices
        self.q_indices[q1_id], self.q_indices[q2_id] = self.q_indices[q2_id], self.q_indices[q1_id]
        # update the number of conflicts
        self.num_conflicts = self.diagonal_conflict_calculator()
        return
    

    def single_queen_conflict_calculator_q1(self, q1, flag) -> tuple:
        """ This function calculates the number of conflicts for a single queen."""
        conflicts = 0
        for i in range(self.N):
            if i != q1:
                if abs(self.q_coordinates[q1][0]-self.q_coordinates[i][0]) == abs(self.q_coordinates[q1][1]-self.q_coordinates[i][1]):
                    conflicts += 1

                    if flag == "old_queen_position":
                        self.conflicting_queens[i] -= 1
                        self.conflicting_queens[q1] -= 1
                        if self.conflicting_queens[i] == 0:
                            del self.conflicting_queens[i]
                            self.non_conflicting_queens.update({i:True})
                        if self.conflicting_queens[q1] == 0:
                            del self.conflicting_queens[q1]
                            self.non_conflicting_queens.update({q1:True}) 

                    elif flag == "new_queen_position":
                        if i in self.conflicting_queens:
                            self.conflicting_queens[i] += 1
                        else:
                            del self.non_conflicting_queens[i]
                            self.conflicting_queens.update({i:1})

                        if q1 in self.conflicting_queens:
                            self.conflicting_queens[q1] += 1
                        else:
                            del self.non_conflicting_queens[q1]
                            self.conflicting_queens.update({q1:1})
                    else:
                            print("ERROR in function single_queen_conflict_calculator")

        return (conflicts)

    def single_queen_conflict_calculator_q2(self, q1, q2,flag) -> tuple:
        """ This function calculates the number of conflicts for a single queen."""
        conflicts = 0
        for i in range(self.N):
            if i != q2:
                if abs(self.q_coordinates[q2][0]-self.q_coordinates[i][0]) == abs(self.q_coordinates[q2][1]-self.q_coordinates[i][1]):
                    conflicts += 1
                    if i != q1:
                        if flag == "old_queen_position":
                            self.conflicting_queens[i] -= 1
                            self.conflicting_queens[q2] -= 1
                            if self.conflicting_queens[i] == 0:
                                del self.conflicting_queens[i]
                                self.non_conflicting_queens.update({i:True})
                            if self.conflicting_queens[q2] == 0:
                                del self.conflicting_queens[q2]
                                self.non_conflicting_queens.update({q2:True}) 

                        elif flag == "new_queen_position":
                            if i in self.conflicting_queens:
                                self.conflicting_queens[i] += 1
                            else:
                                del self.non_conflicting_queens[i]
                                self.conflicting_queens.update({i:1})

                            if q2 in self.conflicting_queens:
                                self.conflicting_queens[q2] += 1
                            else:
                                del self.non_conflicting_queens[q2]
                                self.conflicting_queens.update({q2:1})
                        else:
                            print("ERROR in function single_queen_conflict_calculator")

        return (conflicts)

    def diagonal_conflict_calculator(self) -> int:
        """ This function calculates the total number of conflicts on the board."""
        return sum([self.single_queen_conflict_calculator(_)[0] for _ in range(self.N)])
    
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
                    if self.single_queen_conflict_calculator(qid)[0] > 0:
                        print(bcolors.FAIL + '1' + bcolors.ENDC, end=' ')
                    else:
                        print(bcolors.OKGREEN + '1' + bcolors.ENDC, end=' ')
                    qid += 1
                else:
                    print("0", end=" ")
            print()
        return
    
    def reupdate_1(self,q1,q2):

        for i in range(self.N):
            if i != q1:
                if abs(self.q_coordinates[q1][0]-self.q_coordinates[i][0]) == abs(self.q_coordinates[q1][1]-self.q_coordinates[i][1]):
                    self.conflicting_queens[i] -= 1
                    self.conflicting_queens[q1] -= 1
                    if self.conflicting_queens[i] == 0:
                        del self.conflicting_queens[i]
                        self.non_conflicting_queens.update({i:True})
                    if self.conflicting_queens[q1] == 0:
                        del self.conflicting_queens[q1]
                        self.non_conflicting_queens.update({q1:True})

        for i in range(self.N):
            if i != q2 and i != q1:
                if abs(self.q_coordinates[q2][0]-self.q_coordinates[i][0]) == abs(self.q_coordinates[q2][1]-self.q_coordinates[i][1]):
                    self.conflicting_queens[i] -= 1
                    self.conflicting_queens[q2] -= 1
                    if self.conflicting_queens[i] == 0:
                        del self.conflicting_queens[i]
                        self.non_conflicting_queens.update({i:True})
                    if self.conflicting_queens[q2] == 0:
                        del self.conflicting_queens[q2]
                        self.non_conflicting_queens.update({q2:True})

    def reupdate_2(self, q1, q2):

        for i in range(self.N):
            if i != q1:
                if abs(self.q_coordinates[q1][0]-self.q_coordinates[i][0]) == abs(self.q_coordinates[q1][1]-self.q_coordinates[i][1]):
                    if i not in self.conflicting_queens:
                        del self.non_conflicting_queens[i]
                        self.conflicting_queens.update({i:1})
                    else:
                        self.conflicting_queens[i] += 1
                
                    if q1 not in self.conflicting_queens:
                        del self.non_conflicting_queens[q1]
                        self.conflicting_queens.update({q1:1})
                    else:
                        self.conflicting_queens[q1] += 1

        for i in range(self.N):
            if i != q1 and i != q2:
                if abs(self.q_coordinates[q2][0]-self.q_coordinates[i][0]) == abs(self.q_coordinates[q2][1]-self.q_coordinates[i][1]):
                    if i not in self.conflicting_queens:
                        del self.non_conflicting_queens[i]
                        self.conflicting_queens.update({i:1})
                    else:
                        self.conflicting_queens[i] += 1
                
                    if q2 not in self.conflicting_queens:
                        del self.non_conflicting_queens[q2]
                        self.conflicting_queens.update({q2:1})
                    else:
                        self.conflicting_queens[q2] += 1

    def move(self):
        # 1) pick uniformly at random a couple of queens
        # 2) calculate the acceptance probability by using the conflict function
        # 3) check if the move has to be done

        p1, p2 = 0.9, 0.99
        q = np.random.uniform(0, 1)
        if q < p1 or len(self.non_conflicting_queens) < 2:
            q1_id, q2_id = np.random.choice(list(self.conflicting_queens.keys()), 2, replace=False)
        elif q < p2:
            q1_id = np.random.choice(list(self.conflicting_queens.keys()))
            q2_id = np.random.choice(list(self.non_conflicting_queens.keys()))
        else:
            q1_id, q2_id = np.random.choice(list(self.non_conflicting_queens.keys()), 2, replace=False)

        r1_old, c1_old = self.q_coordinates[q1_id][0], self.q_coordinates[q1_id][1]
        r2_old, c2_old = self.q_coordinates[q2_id][0], self.q_coordinates[q2_id][1]

        old_conflicts_q1 = self.single_queen_conflict_calculator_q1(q1_id,"old_queen_position")
        old_conflicts_q2 = self.single_queen_conflict_calculator_q2(q1_id, q2_id,"old_queen_position")

        eventual_conflict = self.num_conflicts - 2*(old_conflicts_q1 + old_conflicts_q2)

        r1_new, c1_new = r2_old, c1_old
        r2_new, c2_new = r1_old, c2_old

        self.q_coordinates[q1_id][0], self.q_coordinates[q1_id][1] = r1_new, c1_new
        self.q_coordinates[q2_id][0], self.q_coordinates[q2_id][1] = r2_new, c2_new

        new_conflict_q1 = self.single_queen_conflict_calculator_q1(q1_id,"new_queen_position")
        new_conflict_q2 = self.single_queen_conflict_calculator_q2(q1_id, q2_id,"new_queen_position")

        eventual_conflict += 2*(new_conflict_q1 + new_conflict_q2)
        if -self.beta*(eventual_conflict-self.num_conflicts) > 0:
            a = 1 # acceptance probability
        else:
            a = exp(-self.beta*(eventual_conflict-self.num_conflicts)) # acceptance probability

        if (np.random.uniform() < a):
            # in this case the move is made
            self.num_conflicts = eventual_conflict
        else:
            # the move is not made
            self.reupdate_1(q1_id,q2_id)
            self.q_coordinates[q1_id][0], self.q_coordinates[q1_id][1] = r1_old, c1_old
            self.q_coordinates[q2_id][0], self.q_coordinates[q2_id][1] = r2_old, c2_old
            self.reupdate_2(q1_id,q2_id)
        return


    def simulated_annealing(self) -> None:
        print('- Running simulated annealing algorithm...')
        i = 0
        avg = 0
        while(self.num_conflicts > 0):
            self.move()
            avg += self.num_conflicts
            if i % 100 == 0:
                print(f'Swap {i}, avg conflicts: {avg/1000}')
                avg = 0
            i+=1
        assert self.is_safe()
        return


if __name__ == "__main__":
    board = NQueens(beta = 1, N=1000)
    board.main_diagonal_initialisation()
    print(f'---Board initialisation of size: {board.N}x{board.N}---')
    # print('- Board:')
    # board.display_board()
    print(f'- Total conflicts: {board.num_conflicts}')

    start = time()
    board.simulated_annealing()
    end = time()
    print(f'Runtime : {end-start:.3f} seconds for {board.N} queens')
    
    # print('- Final board:')
    # board.display_board()
    print('Done')
    #print(board.q_coordinates)
