# mutual information enstimation
# get the MI for Y and X_r
# compare to pairwise and bmp

# since the MI for pair calculated with different random weights need to average over iteration
# to speed up, decrease the number of iterations or decrease N_x

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
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.titlesize'] = 14

# np.random.seed(0)

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

def get_kwta(x, w, a_y):
    y = kWTA2(w @ x, a_y)
    return y

N_y = 25
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

a_w = 7
# w2 = generate_random_matrix(N_y, N_x, a_w)
# w_pair = generate_random_matrix(N_y **2, N_x, a_w)
a_y_range = np.arange(1, N_y, 2)
iters = 5

mut_info5_bmp =  np.zeros((iters, a_y_range.size))
mut_info5x =  np.zeros((iters, a_y_range.size))
mut_info5x2 = np.zeros((iters, a_y_range.size))
error = np.zeros((a_y_range.size, X_size))
for iter in range(iters):
    print('New iter')
    print(iter)
    w = generate_random_matrix(N_y, N_x, a_w)
    w_pair = generate_random_matrix(N_y ** 2, N_x, a_w)
    for i, ai in enumerate(a_y_range):
        print(ai)
        Y = np.zeros((X_size, N_y), dtype=int)
        Y_bmp = np.zeros((X_size, N_y), dtype=int)
        X_r = np.zeros((X_size, N_x), dtype=int)
        X_r_bmp = np.zeros((X_size, N_x), dtype=int)
        X_r_pair = np.zeros((X_size, N_x), dtype=int)
        for k, x in enumerate(X):
            Y[k] = kWTA2(w @ x, ai).astype(int)
            Y_bmp[k] = get_omp(x, w, ai).astype(int)
            X_r[k] = kWTA2(w.T @ Y[k], np.count_nonzero(x)).astype(int)
            X_r_bmp[k] = kWTA2(w.T @ Y_bmp[k], np.count_nonzero(x)).astype(int)
            X_r_pair[k] = kWTA2(w_pair.T @ np.outer(Y[k], Y[k]).flatten(), np.count_nonzero(x)).astype(int)

        uni_x, counts_x = np.unique(X_r, return_counts=True, axis=0)
        uni_x_bmp, counts_x_bmp2 = np.unique(X_r_bmp, return_counts=True, axis=0)
        uni_x2, counts_x2 = np.unique(X_r_pair, return_counts=True, axis=0)
        mut_info5x[iter, i] = 1 - np.sum(counts_x * np.log2(counts_x)) / (X_size * N_x )
        mut_info5x2[iter, i] = 1 - np.sum(counts_x2 * np.log2(counts_x2)) / (X_size * N_x )
        mut_info5_bmp[iter, i] = 1 - np.sum(counts_x_bmp2 * np.log2(counts_x_bmp2)) / (X_size * N_x )


plt.plot(a_y_range / N_y,  np.mean(mut_info5_bmp, axis=0), '-o',  markersize=10, label='BMP')
plt.plot(a_y_range / N_y,  np.mean(mut_info5x, axis=0), '-d',  markersize=10, label='kWTA')
plt.plot(a_y_range / N_y, np.mean(mut_info5x2, axis=0), '--x',  markersize=10, label='Pairwise sigma-pi')

plt.ylim([0, 1])
plt.xlim([0, 1])
plt.xlabel(r'$s_y$')
plt.ylabel(r'Scaled mutual information')
plt.legend(loc='lower right')
# plt.title(str(N_x) +' ' + str(N_y))
plt.savefig('figures/mutual_information_pair', bbox_inches='tight')
plt.show()
