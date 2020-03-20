# Intermidiate steps for mutual information enstimation
# how many different x lead to single y. Computational experiments

# see encoding 12 series

# for fixed a_x
# graph kwta vs bmp
# For all 2^N input vectors

# use pairwise corelation



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


def get_omp(x, w, a_y):
    y = np.zeros(w.shape[0])
    r = np.copy(x)
    a_x = np.count_nonzero(x)
    # print(a_x)
    for k in range(a_y):
        z = w @ r
        z[np.nonzero(y)[0]] = -100
        ind = np.argmax(z)
        y[ind] = 1
        # error_kwta2 = np.zeros(a_x_range.size)
        # for j, axi in enumerate(a_x_range):
        #     x_r = kWTA2(w.T @ y, axi)
        #     error_kwta2[j] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
        # a_x_optimal = a_x_range[np.argmin(error_kwta2)]
        # x_r = kWTA2(w.T @ y, a_x_optimal)
        x_r = kWTA2(w.T @ y, a_x)
        r = 2 * x - x_r
    return y

N_y = 22
N_x = 15


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
a_y_range = np.arange(1, N_y, 2)
mut_info4 = np.zeros(a_y_range.size)
mut_info5 = np.zeros(a_y_range.size)
mut_info4bmp = np.zeros(a_y_range.size)
mut_info5bmp = np.zeros(a_y_range.size)
mut_info5pair = np.zeros(a_y_range.size)
mut_info5pair3 = np.zeros(a_y_range.size)
a_w = 5
w_pair = generate_random_matrix(N_y, N_x ** 2, a_w ** 2)
# w_pair3 = generate_random_matrix(N_y, N_x ** 3, a_w ** 3)

for i, ai in enumerate(a_y_range):
    print(ai)
    Y = np.zeros((X_size, N_y), dtype=int)
    Y_bmp = np.zeros((X_size, N_y), dtype=int)
    Y_pair = np.zeros((X_size, N_y), dtype=int)
    # Y_pair3 = np.zeros((X_size, N_y), dtype=int)
    for k, x in enumerate(X):
        Y[k] = kWTA2(w @ x, ai).astype(int)
        Y_bmp[k] = get_omp(x, w, ai).astype(int)
        Y_pair[k] = kWTA2(w_pair @ np.outer(x, x).flatten(), ai).astype(int)
        # Y_pair3[k] = kWTA2(w_pair3 @ np.outer(np.outer(x, x).flatten(), x).flatten(), ai).astype(int)
    uni, counts = np.unique(Y, return_counts=True, axis=0)
    uni2, counts2 = np.unique(Y_bmp, return_counts=True, axis=0)
    uni3, counts3 = np.unique(Y_pair, return_counts=True, axis=0)
    # uni4, counts4 = np.unique(Y_pair3, return_counts=True, axis=0)
    # mut_info4[i] = 1 - uni.shape[0] * np.mean(counts) * np.log2(np.mean(counts)) / ( X_size * N_x )
    mut_info5[i] = 1 - np.sum(counts * np.log2(counts)) / ( X_size * N_x )
    # mut_info4bmp[i] = 1 - uni2.shape[0] * np.mean(counts2) * np.log2(np.mean(counts2)) / (X_size * N_x)
    mut_info5bmp[i] = 1 - np.sum(counts2 * np.log2(counts2)) / (X_size * N_x)
    mut_info5pair[i] = 1 - np.sum(counts3 * np.log2(counts3)) / (X_size * N_x)
    # mut_info5pair3[i] = 1 - np.sum(counts4 * np.log2(counts4)) / (X_size * N_x)


    # plt.hist(counts)
    # plt.show()

print(a_y_range / N_y)
print(mut_info4)
print(mut_info5)
print(mut_info4bmp)
print(mut_info5bmp)
print(mut_info5pair)

# plt.plot(a_y_range / N_y, mut_info4, '--o', label=r'Approximated, $\Omega^*$ ')
plt.plot(a_y_range / N_y, mut_info5, '-o', label='Precise kwta')
# plt.plot(a_y_range / N_y, mut_info4bmp, 'r--o', label=r'Approximated, $\Omega^*$  bmp')
plt.plot(a_y_range / N_y, mut_info5bmp, 'r-+', label='Precise bmp')
plt.plot(a_y_range / N_y, mut_info5pair, 'b-d', label='Precise pair')
# plt.plot(a_y_range / N_y, mut_info5pair3, 'c-o', label='Precise pair 3')
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.xlabel(r'$s_y$')
plt.ylabel(r'Scaled mutual information')
plt.legend()
plt.savefig('figures/mutual_information_pair', bbox_inches='tight')
plt.show()
