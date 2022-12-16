import numpy as np
from time import time
from NQueens_test import NQueens
from math import log, exp

# APPROXIMATION TO NB OF VALID SOLUTIONS = (0.143 * N)^N


# STEPS TO DO
# 1. Fix a number of iterations
# 2. Run the Metropolis algorithm for multiple values of beta between 0 and some beta_T
# 3. When the algorithm stops for some beta_i, select i to be the new T
# 4. For every beta_i with some fixed step, sample the M (large enough) X_i's according to our init distribution
# 5. Compute the average of the X_i's and repeat the procedure till we reach beta_infty.
# 6. The approximate number of solutions should be close enough to beta_infty.

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
        self.debug_flag = False

    # This property must be used to place queens on the board
    def is_safe(self) -> bool:
        """ This function asserts that there exists no row or column with more than one queen on it."""
        for i in range(self.N):
            if sum(self.q_coordinates[:, 0] == i) > 1 or sum(self.q_coordinates[:, 1] == i) > 1:
                return False
        return True
    

    def random_positions_initialisation(self) -> tuple:
        """ This function randomly puts queens on the board, whilst preserving the no row/column conflict contraint. """
        a, b = np.arange(self.N), np.arange(self.N)
        np.random.shuffle(a)
        np.random.shuffle(b)
        self.q_coordinates = np.array([(a[i], b[i]) for i in range(self.N)])
        self.q_indices = np.array([a[i] + i*self.N for i in range(self.N)])
        self.num_conflicts = self.diagonal_conflict_calculator()
        self.conflicting_queens = {_:self.single_queen_initial_conflicts(_) for _ in range(self.N)}
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
    

    def single_queen_initial_conflicts(self, queen_id) -> int:
        """ This function calculates the number of conflicts for a single queen."""
        conflicts = 0
        for i in range(self.N):
            if i != queen_id:
                if abs(self.q_coordinates[queen_id][0]-self.q_coordinates[i][0]) == abs(self.q_coordinates[queen_id][1]-self.q_coordinates[i][1]):
                    conflicts += 1
        return conflicts
    

    def diagonal_conflict_calculator(self) -> int:
        """ This function calculates the total number of conflicts on the board."""
        # sum conflicts with numpy
        return sum([self.single_queen_initial_conflicts(_) for _ in range(self.N)])
    

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

    
    def reupdate_1(self,q1,q2):

        for i in range(self.N):
            if i != q1:
                if abs(self.q_coordinates[q1][0]-self.q_coordinates[i][0]) == abs(self.q_coordinates[q1][1]-self.q_coordinates[i][1]):
                    self.conflicting_queens[i] -= 1
                    self.conflicting_queens[q1] -= 1
                    assert self.conflicting_queens[i] >= 0
                    assert self.conflicting_queens[q1] >= 0
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
                    assert self.conflicting_queens[i] >= 0
                    assert self.conflicting_queens[q2] >= 0
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
        p1, p2 = 0.8, 0.9
        q = np.random.uniform(0, 1)
        if len(self.conflicting_queens) == 1:
            curr_qid = list(self.conflicting_queens.keys())[0]
            del self.conflicting_queens[curr_qid]
            self.non_conflicting_queens.update({curr_qid:True})
        if len(self.conflicting_queens) >= 2 and len(self.non_conflicting_queens) >= 2:
            if q < p1:
                q1_id, q2_id = np.random.choice(list(self.conflicting_queens.keys()), 2, replace=False)
                prob_xy = p1
            elif q < p2:
                q1_id = np.random.choice(list(self.conflicting_queens.keys()))
                q2_id = np.random.choice(list(self.non_conflicting_queens.keys()))
                prob_xy = p2-p1
            else:
                q1_id, q2_id = np.random.choice(list(self.non_conflicting_queens.keys()), 2, replace=False)
                prob_xy = 1-p2
        else:
            if len(self.conflicting_queens) == 0:
                q1_id, q2_id = np.random.choice(list(self.non_conflicting_queens.keys()), 2, replace=False)
            else:
                q1_id,q2_id = np.random.choice(list(self.conflicting_queens.keys()), 2, replace=False)
            prob_xy = 1

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

        if len(self.conflicting_queens) >= 2 and len(self.non_conflicting_queens) >= 2:
            if q1_id in self.conflicting_queens and q2_id in self.conflicting_queens:
                prob_yx = p1
            elif q1_id in self.non_conflicting_queens and q2_id in self.non_conflicting_queens:
                prob_yx = 1-p2
            else:
                prob_yx = p2-p1
        else:
            prob_yx = 1

        eventual_conflict += 2*(new_conflict_q1 + new_conflict_q2)
        if -self.beta*(eventual_conflict-self.num_conflicts)+ log(prob_yx/prob_xy)> 0:
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
        if self.num_conflicts == 0:
            self.debug_flag = True
        return


def fact(n):
    if n == 0:
        return 1
    else:
        return n * fact(n-1)


def create_queen_solutions_dict():
    queen_dict = {}
    with open("queen_dict.txt", "r") as f:
        for line in f:
            (key, val) = line.split(', ')
            queen_dict[int(key)] = int(val)
    return queen_dict


if __name__ == "__main__":
    MAX_T = 2.0
    NUM_QUEENS = 12
    NUM_ITERATIONS_CHAIN = 4000
    SUPP_ITERATIONS = int(NUM_ITERATIONS_CHAIN / 10)
    M = 200
    delta_beta = 0.5

    ratios = [] #list of ratios
    betas = [0]
    flag_beta = True
    i = 0
    num_solutions = fact(NUM_QUEENS)
    #iterate through every beta until beta_star is found (and therefore flag_beta becomes false)
    while flag_beta:
        j = 0
        somma = 0
        print("Started calculations for beta =",betas[i])
        #run the metropolis algo M times: the greater M the better the approximation (central limit theorem)
        while(j < M):
            #print("Trial",j)
            board = NQueens(beta = betas[i], N=NUM_QUEENS)
            board.random_positions_initialisation()
            k = 0
            #run the chain for NUM_ITERATIONS_CHAIN times
            while(k < NUM_ITERATIONS_CHAIN):
                board.move()
                k += 1
            somma += exp(-delta_beta*board.num_conflicts)
            j += 1
        somma /= M
        ratios.append(somma)
        num_solutions *= somma
        print("beta =", betas[i], "total number of solutions =", num_solutions)
        # if at least 90% of the M times the chain had at least 95% of zero cost function, then we reached beta_star
        if num_solutions < 5000:
            flag_beta = False
        else:
            betas.append(betas[i]+delta_beta)
            i += 1
    
    print("betas =",betas)
    print("ratios =", ratios)
    print("solutions =", num_solutions)

    print('Done')
