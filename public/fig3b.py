# Intermidiate steps for mutual information enstimation
# how many different x lead to single y. Computational experiments

# see encoding 12 series

# for fixed a_x
# graph kwta
# For all 2^N input vectors

# get the count of unique y vectors

#paper graph

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import itertools
import matplotlib as mpl


mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

mpl.rcParams['figure.figsize'] = [10.0, 8.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 800
mpl.rcParams['savefig.format'] = 'pdf'

mpl.rcParams['font.size'] = 24
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['figure.titlesize'] = 14

# np.random.seed(0)
# print(comb(20, 5))
# quit()

########################
def kWTA2(cells, k):
    # n_active = max(int(sparsity * cells.size), 1)
    winners = np.argsort(cells)[-k:]
    sdr = np.zeros(cells.shape, dtype=cells.dtype)
    sdr[winners] = 1
    return sdr

def generate_random_vector(N, a_x):
    vector = np.zeros(N, dtype=int)
    ones = np.random.choice(N, size=a_x, replace=False)
    vector[ones] = 1
    return vector

def generate_random_matrix(R, N, a_x):
    matrix = np.zeros((R, N), dtype=int)
    for i in range(R):
        matrix[i] = generate_random_vector(N, a_x)
    return matrix


def get_kwta(x, w, a_y):
    y = kWTA2(w @ x, a_y)
    return y

N_y = 30
N_x = 20


x_space = []
stuff = np.arange(N_x)
for L in range(0, stuff.size + 1):
    for subset in itertools.combinations(stuff, L):
        x_space.append(list(subset))

x_space.pop(0)
X_size = len(x_space)
X = np.zeros((X_size, N_x), dtype=int)
print(X_size)
for i, inds in enumerate(x_space):
    # print(inds)
    X[i, inds] = 1

# print(X[15])

# quit()
a_w = 5
w = generate_random_matrix(N_y, N_x, a_w)
a_y_range = np.arange(1, N_y, 1)
mut_info4 = np.zeros(a_y_range.size)
mut_info5 = np.zeros(a_y_range.size)
for i, ai in enumerate(a_y_range):
    print(ai)
    Y = np.zeros((X_size, N_y), dtype=int)
    for k, x in enumerate(X):
        Y[k] = kWTA2(w @ x, ai).astype(int)
    uni, counts = np.unique(Y, return_counts=True, axis=0)
    mut_info4[i] = 1 - uni.shape[0] * np.mean(counts) * np.log2(np.mean(counts)) / ( X_size * N_x )
    mut_info5[i] = 1 - np.sum(counts * np.log2(counts)) / ( X_size * N_x )
    # plt.hist(counts)
    # plt.show()

print(a_y_range / N_y)
print(mut_info4)
print(mut_info5)

plt.plot(a_y_range / N_y, mut_info4, '--o', label=r'Approximated, $\Omega^*$ ')
plt.plot(a_y_range / N_y, mut_info5, '-o', label='Precise')
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.xlabel(r'$s_y$')
plt.ylabel(r'Scaled mutual information')
plt.legend()
plt.savefig('figures/mutual_information', bbox_inches='tight')
plt.show()
