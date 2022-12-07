'''
This is the file for stress testing our algorithm through various iterations.

@todo
1) write visualisation tools methods
2) optimise parameter testing -> utilise multiple cores and do it in parallel
'''

from time import time
from typing import List

from NQueens import NQueens
import matplotlib.pyplot as plt
import numpy as np

def visualise(num_queens_lst: List[int], lst_of_times: List[List[float]]):
    # plot lines
    for lst_time in lst_of_times:
        plt.plot(lst_time,num_queens_lst)
    plt.legend()
    plt.show()

def performance_testing():
    # set_of_queens = [10,20,30,40,50,60,70,80,90,100]
    set_of_queens = [10,20,30]

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
        diagonal_initial_times.append(avg_runtime)
    '''
    Calculating for the knight initial positioning!
    '''
    knight_initial_times = []
    for num_queens in set_of_queens:
        avg_runtime = performance_analysis("knight",beta=1,num_queens=num_queens)
        knight_initial_times.append(avg_runtime)


    print('The tests were ran for the setting of ' + str(set_of_queens) + " queens.")
    print()
    print('For Random Board Initialisation:')
    for res in random_initial_times:
        print(res)
    print()
    print('For Knight Board Initialisation:')
    for res in knight_initial_times:
        print(res)
    print()
    print('For Diagonal Board Initialisation:')
    for res in diagonal_initial_times:
        print(res)
    visualise(set_of_queens,[random_initial_times,diagonal_initial_times,knight_initial_times])


    # return random_initial_times,diagonal_initial_times,knight_initial_times




def performance_analysis(initial_positioning:str, beta:int, num_queens:int):
    '''
    @param  initial_positioning : ["random","diagonal","knight"]
    @param  beta = whatever number your heart desires
    @param  num_queens = whatever number your heart desires
    '''
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
    # performance_analysis(initial_positioning="random",beta=1,num_queens=100)

    performance_testing()

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
