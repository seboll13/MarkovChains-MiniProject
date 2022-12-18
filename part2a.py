import numpy as np
from math import exp, log
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm

list_regine = [8, 12, 16, 23, 26]
lista_beta = [3.5, 11.5, 22.5, 37, 45.5]

pendenza, quota = np.polyfit(list_regine,lista_beta,1)

x = []
y = []
for num in range(8,51):
    x.append(num)
    y.append(num*pendenza+quota)

plt.plot(list_regine,lista_beta,'--ro',x,y,'b')
print(x)
print(y[35-8])
plt.show()