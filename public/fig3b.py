# mutual information enstimation
# For all 2^N input vectors
# get the MI for Y and X_r


import numpy as np
import matplotlib.pyplot as plt
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


########################
def kWTA2(cells, k):
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


a_w = 7
w = generate_random_matrix(N_y, N_x, a_w)
w2 = generate_random_matrix(N_y, N_x, a_w)
a_y_range = np.arange(1, N_y, 2)
mut_info4 = np.zeros(a_y_range.size)
mut_info5 = np.zeros(a_y_range.size)
mut_info4x = np.zeros(a_y_range.size)
mut_info5x = np.zeros(a_y_range.size)
error = np.zeros((a_y_range.size, X_size))
for i, ai in enumerate(a_y_range):
    print(ai)
    Y = np.zeros((X_size, N_y), dtype=int)
    X_r = np.zeros((X_size, N_x), dtype=int)
    for k, x in enumerate(X):
        Y[k] = kWTA2(w @ x, ai).astype(int)
        X_r[k] = kWTA2(w.T @ Y[k], np.count_nonzero(x)).astype(int)
        # X_r[k] = kWTA2(w2.T @ Y[k], np.count_nonzero(x)).astype(int)  # shows that for random decoder weight the MI the same
        error[i, k] = np.sum(np.abs(x - X_r[k]))
    uni, counts = np.unique(Y, return_counts=True, axis=0)
    uni_x, counts_x = np.unique(X_r, return_counts=True, axis=0)
    mut_info4[i] = 1 - uni.shape[0] * np.mean(counts) * np.log2(np.mean(counts)) / ( X_size * N_x )
    mut_info5[i] = 1 - np.sum(counts * np.log2(counts)) / ( X_size * N_x )
    mut_info5x[i] = 1 - np.sum(counts_x * np.log2(counts_x)) / (X_size * N_x )

# print(a_y_range / N_y)
# print(mut_info4)
# print(mut_info5)

plt.plot(a_y_range / N_y, mut_info4, '--o', label=r'I(X, Y) upper bound')
plt.plot(a_y_range / N_y, mut_info5, '-o', label='I(X, Y)')
plt.plot(a_y_range / N_y, mut_info5x, '-d',  markersize=10, label=r'$I(X,X_r)=I(Y,X_r)$')
plt.plot(a_y_range / N_y, np.mean(error, axis=1) / N_x, '-*', markersize=10, label=r'Error')
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.xlabel(r'$s_y$')
plt.ylabel(r'Scaled mutual information')
plt.legend(loc='lower right')
# plt.savefig('figures/mutual_information', bbox_inches='tight')
plt.show()
