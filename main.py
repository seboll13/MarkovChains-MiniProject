import numpy as np
from math import exp, floor

class states_fn():
    """ The states that represent the queens positions on the NxN chessboard
    @param beta: parameter used during the algorithm
    @param N: number of rows and columns of the chessboard
    @param coords: list of arrays - every array is 2-dimensional and contains the coordinates
    of a queen; 'states' contains N items in total, one per queen
    @param indices: list of integers - every integer is the index of the queen in the 1-dimensional
    array that represents the chessboard
    @param id_chosen: integer - id of the queen that will be moved
    """
    def __init__(self, beta=1, N=8, coords=[], indices=[], id_chosen=0):
        self.beta = beta
        self.N = N
        self.coords = coords
        self.indices = indices
        self.id_chosen = id_chosen
        # 'conflict' contains the current value of the function that has to be minimized
        self.conflict = self.N*(self.N-1)
    
    
    def initialize_chain(self):
        coord = []
        indices = []
        for ii in range(self.N):
            index = ii*self.N+ii
            coord.append(np.array([ii,ii]))
            indices.append(index)
        return coord, indices


    def conflict_calc(self,x,y):
        conflitto = self.conflict
        for ii in range(self.N):
            if ii != self.id_chosen:
                if self.coords[self.id_chosen][0] == self.coords[ii][0]:
                    conflitto -= 2
                elif self.coords[self.id_chosen][1] == self.coords[ii][1]:
                    conflitto -= 2
                elif abs(self.coords[self.id_chosen][0] - self.coords[ii][0]) == abs(self.coords[self.id_chosen][1] - self.coords[ii][1]):
                    conflitto -= 2
                
                if x == self.coords[ii][0]:
                    conflitto += 2
                elif y == self.coords[ii][1]:
                    conflitto += 2
                elif abs(x - self.coords[ii][0]) == abs(y - self.coords[ii][1]):
                    conflitto += 2
        return conflitto


    def next_move(self):
        """ The following function finds the next queen to be moved and the x,y coordinates of the move of the chosen queen. It also updates the states after the move.
        """
        self.id_chosen = np.random.randint(0,self.N)
        p = np.zeros(self.N**2) # prob distribution
        index = 0
        psi = 1/(self.N**2-self.N)# prob of the base chain: uniform
        for ii in range(self.N):
            for jj in range(self.N):
                if index != self.indices[self.id_chosen]: # check to be not over the chosen queen 
                    if index in self.indices: # check if we are over some queen
                        p[index] = 0
                    else:
                        eventual_confl = self.conflict_calc(ii,jj)
                        var = exp(-self.beta*(eventual_confl-self.conflict))
                        a = min(1,var)
                        p[index] = a*psi
                index += 1
        p[self.indices[self.id_chosen]] = 1-np.sum(p)
        index_next_move = np.random.choice(self.N**2,p = p)
        
        x_next_move = floor(index_next_move/self.N)
        y_next_move = index_next_move%self.N

        self.conflict = self.conflict_calc(x_next_move,y_next_move)
        print(self.conflict)
        self.coords[self.id_chosen] = np.array([x_next_move,y_next_move])
        self.indices[self.id_chosen] = x_next_move*self.N + y_next_move
    

    def next_move_2(self):
        """ The following function finds the next queen to be moved and the x,y coordinates of the move of the chosen queen. It also updates the states after the move."""
        cnt_nb = 0 # number of neighbor queens: it can be at most 2
        right_nb = -1
        left_nb = -1
        while cnt_nb == 0:
            id = np.random.randint(0,self.N)
            if (self.indices[id]+1 not in self.indices) and self.indices[id]+1 < self.N**2:
                cnt_nb += 1
                right_nb = self.indices[id]+1
            if (self.indices[id]-1 not in self.indices) and self.indices[id]-1 > -1:
                cnt_nb += 1
                left_nb = self.indices[id]-1

        self.id_chosen = id
        if cnt_nb == 1:
            if right_nb != -1:
                x = floor(right_nb/self.N)
                y = right_nb%self.N
                eventual_confl = self.conflict_calc(x,y)
                var = exp(-self.beta*(eventual_confl-self.conflict))
                a = min(1,var)
                index_next_move = np.random.choice([right_nb-1,right_nb],p = [1-a,a])
            else:
                x = floor(left_nb/self.N)
                y = left_nb%self.N
                eventual_confl = self.conflict_calc(x,y)
                var = exp(-self.beta*(eventual_confl-self.conflict))
                a = min(1,var)
                index_next_move = np.random.choice([left_nb,left_nb+1],p = [a,1-a])
        else:
            x = floor(right_nb/self.N)
            y = right_nb%self.N
            eventual_confl = self.conflict_calc(x,y)
            var = exp(-self.beta*(eventual_confl-self.conflict))
            a_right = min(1,var)

            x = floor(left_nb/self.N)
            y = left_nb%self.N
            eventual_confl = self.conflict_calc(x,y)
            var = exp(-self.beta*(eventual_confl-self.conflict))
            a_left = min(1,var)
            index_next_move = np.random.choice([left_nb,left_nb+1,right_nb],p = [0.5*a_left, 1-0.5*(a_left+a_right),0.5*a_right])
        
        x_next_move = floor(index_next_move/self.N)
        y_next_move = index_next_move%self.N

        self.conflict = self.conflict_calc(x_next_move,y_next_move)
        print(self.conflict)
        self.coords[self.id_chosen] = np.array([x_next_move,y_next_move])
        self.indices[self.id_chosen] = x_next_move*self.N + y_next_move


    def next_move_3(self):
        self.id_chosen = np.random.randint(0,self.N) # pick uniformly at random the id of the queen that will be moved
        p = np.zeros(self.N) # array of the metropolis chain probabilities
        psi = 1/self.N # uniform probability of the base chain 

        for ii in range(self.N):
            if self.coords[self.id_chosen][1] != ii: # check that next state is not the previous one
                eventual_confl = self.conflict_calc(self.coords[self.id_chosen][0],ii) # conflict of the next state
                a = min(1,exp(-self.beta*(eventual_confl-self.conflict))) # acceptance probability 
                p[ii] = psi*a # metropolis chain probability to go to the next state

        p[self.coords[self.id_chosen][1]] = 1-np.sum(p)

        next_column = np.random.choice(self.N,p = p) # now find the next move

        self.conflict = self.conflict_calc(self.coords[self.id_chosen][0],next_column)
        print(self.conflict)
        self.coords[self.id_chosen] = np.array([self.coords[self.id_chosen][0],next_column])
        self.indices[self.id_chosen] = self.coords[self.id_chosen][0]*self.N + next_column


if __name__ == '__main__':            
    _class = states_fn(beta=1.5, N=100)
    _class.coords, _class.indices = _class.initialize_chain()
    while _class.conflict != 0:
        _class.next_move_3()
    print(_class.coords)
