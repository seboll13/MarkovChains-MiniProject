'''
This is the file for stress testing our algorithm through various iterations.
'''

import numpy as np
from math import exp
from time import time
from NQueens import NQueens

def performance_testing():
    set_of_queens = [10,20,30,40,50,60,70,80,90,100]

    '''
    Calculating for the random initial positioning!
    '''
    random_initial_times = []
    for num_queens in set_of_queens:
        avg_runtime = performance_analysis("random",beta=1,num_queens=num_queens)
        random_initial_times.append(avg_runtime)
    '''
    Calculating for the diagonal initial positioning!
    '''
    diagonal_initial_times = []
    for num_queens in set_of_queens:
        avg_runtime = performance_analysis("diagonal",beta=1,num_queens=num_queens)
        random_initial_times.append(avg_runtime)
    '''
    Calculating for the knight initial positioning!
    '''
    knight_initial_times = []
    for num_queens in set_of_queens:
        avg_runtime = performance_analysis("knight",beta=1,num_queens=num_queens)
        random_initial_times.append(avg_runtime)



'''
initial_positioning = ["random","diagonal","knight"]
beta = whatever number your heart desires
num_queens = whatever number your heart desires
'''
def performance_analysis(initial_positioning:str, beta:int, num_queens:int):
    if initial_positioning == "random":
        board = NQueens(beta=beta, N=num_queens)
        board.random_positions_initialisation()

    elif initial_positioning == "diagonal":
        board = NQueens(beta=beta, N=num_queens)
        board.main_diagonal_initialisation()

    elif initial_positioning == "knight":
        board = NQueens(beta=beta, N=num_queens)
        board.knight_initialisation()

    print(f'---Board initialisation of size: {board.N}x{board.N}---')
    print(f'- Total conflicts: {board.num_conflicts}')

    avg_runtime = 0
    times = []
    for _ in range(5):
        print(f'Start of iteration {_ + 1}')
        board.single_queen_conflict_calculator(0)
        start = time()
        board.simulated_annealing()
        end = time()
        elapsed = end - start
        times.append(elapsed)
        avg_runtime += elapsed
        board.random_positions_initialisation()
    print(f'Average runtime: {avg_runtime / 5:.3f} seconds for {board.N} queens')
    print('Done')

    # print(times)
    return avg_runtime

if __name__ == '__main__':
    performance_analysis(initial_positioning="random",beta=1,num_queens=100)
# board = NQueens(beta = 1, N=100)
# board.random_positions_initialisation()
#
# print(f'---Board initialisation of size: {board.N}x{board.N}---')
# print(f'- Total conflicts: {board.num_conflicts}')
#
# avg_runtime = 0
# for _ in range(5):
#     print(f'Start of iteration {_+1}')
#     board.single_queen_conflict_calculator(0)
#     start = time()
#     board.simulated_annealing()
#     end = time()
#     elapsed = end-start
#     avg_runtime += elapsed
#     board.random_positions_initialisation()
# print(f'Average runtime: {avg_runtime/5:.3f} seconds for {board.N} queens')
# print('Done')
