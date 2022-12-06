'''
This is the file for stress testing our algorithm through various iterations.
'''

import numpy as np
from math import exp
from time import time
from NQueens import NQueens

if __name__ == '__main__':
    board = NQueens(beta = 1, N=100)
    board.main_diagonal_initialisation()

    print(f'---Board initialisation of size: {board.N}x{board.N}---')
    print(f'- Total conflicts: {board.num_conflicts}')

    avg_runtime = 0
    for _ in range(5):
        print(f'Start of iteration {_+1}')
        board.single_queen_conflict_calculator(0)
        start = time()
        board.simulated_annealing()
        end = time()
        elapsed = end-start
        avg_runtime += elapsed
        board.main_diagonal_initialisation()
    print(f'Average runtime: {avg_runtime/5:.3f} seconds for {board.N} queens')
    print('Done')

