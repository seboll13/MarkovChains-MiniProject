import numpy as np
from time import time
from NQueens_test import NQueens

MAX_T = 2.0
NUM_QUEENS = 100
NUM_ITERATIONS = 10

# APPROXIMATION TO NB OF VALID SOLUTIONS = (0.143 * N)^N


# STEPS TO DO
# 1. Fix a number of iterations
# 2. Run the Metropolis algorithm for multiple values of beta between 0 and some beta_T
# 3. When the algorithm stops for some beta_i, select i to be the new T
# 4. For every beta_i with some fixed step, sample the M (large enough) X_i's according to our init distribution
# 5. Compute the average of the X_i's and repeat the procedure till we reach beta_infty.
# 6. The approximate number of solutions should be close enough to beta_infty.

if __name__ == '__main__':
    for beta in np.arange(1.0, MAX_T, 0.1):
        avg_times = 0
        for _ in range(NUM_ITERATIONS):
            board = NQueens(beta=beta, N=NUM_QUEENS)
            board.main_diagonal_initialisation()
            start = time()
            board.simulated_annealing(False)
            end = time()
            avg_times += (end - start)
        avg_times /= NUM_ITERATIONS
        print(f'Run with beta={beta} took {avg_times:.3f} seconds on avg.')
    print('Done')