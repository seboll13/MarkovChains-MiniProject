'''
This is the file for stress testing our algorithm through various iterations.

@todo
1) write visualisation tools methods
2) optimise parameter testing -> utilise multiple cores and do it in parallel
'''

from time import time

from NQueens_test import NQueens
import numpy as np


def write_results(file, init_pos, beta, avg_time, std_time, avg_iterations, std_iterations) -> None:
    with open(file, 'a') as f:
        f.write(f'beta = {beta} ; initial positioning = {init_pos}\n')
        f.write(f'Average elapsed time = {avg_time} seconds with std = {std_time} seconds.\n')
        f.write(f'Average number of iterations = {avg_iterations} with std = {std_iterations}.\n')
        f.write('#' * 30 + '\n')
    return


def performance_analysis(initial_positioning:str, beta:int, num_queens:int):
    NUM_RUNS = 10
    iterations = np.zeros(NUM_RUNS)
    runtimes = np.zeros(NUM_RUNS)

    print(f'Starting performance analysis for {NUM_RUNS} runs with beta = {beta} and {initial_positioning} initial positioning...')
    for _ in range(NUM_RUNS):
        board = NQueens(beta=beta, N=num_queens)
        if initial_positioning == "diagonal":
            board.main_diagonal_initialisation()
        elif initial_positioning == "knight":
            board.knight_initialisation()
        else: # random case
            board.random_positions_initialisation()
        
        start = time()
        num_iterations = board.simulated_annealing(False)
        end = time()

        if num_iterations is not None:
            iterations[_] = num_iterations
        print(f'Iteration {_+1} => runtime of {end-start:.3f} seconds.')
        runtimes[_] = end-start
    
    avg_time = round(np.mean(runtimes), 3)
    std_time = round(np.std(runtimes), 3)
    avg_iterations = round(np.mean(iterations), 3)
    std_iterations = round(np.std(iterations), 3)
    print(f"Average elapsed time = {avg_time} seconds")
    print(f"Standard deviation of elapsed time = {std_time} seconds")
    print(f"Average number of iterations = {avg_iterations}")
    print(f"Standard deviation of number of iterations = {std_iterations}")

    write_results('results.txt', initial_positioning, beta, avg_time, std_time, avg_iterations, std_iterations)
    print(f'#' * 30 + '')
    return


# TODO: handle plots later


if __name__ == '__main__':
    N = 1000 # number of queens
    betas = [1.0, 1.5, 2.0, 2.5, 3.0]
    initial_positionings = ["diagonal", "knight", "random"]

    for beta in betas:
        for pos in initial_positionings:
            performance_analysis(initial_positioning=pos, beta=beta, num_queens=N)
