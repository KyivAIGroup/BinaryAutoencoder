# mutual information enstimation
# get the MI for Y and X_r for special matrices


import numpy as np
import matplotlib.pyplot as plt
# from scipy.special import comb
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



def generate_random_matrix2(R, N, a_x):
    w = generate_random_matrix(R, N, a_x)
    mean_col = np.mean(np.count_nonzero(w, axis=0))
    non_zeros = np.count_nonzero(w, axis=0)
    non_zeros_sor = np.argsort(np.count_nonzero(w, axis=0))[::-1]
    # print(non_zeros_sor)
    # print(non_zeros[non_zeros_sor])
    for i, mi in enumerate(non_zeros_sor):
        # print(i, mi, np.count_nonzero(w[:, mi]))
        # if non_zeros[mi] > mean_col:
        # print(non_zeros[mi], mean_col)
        for k in range(int(np.count_nonzero(w[:, mi]) - mean_col)):
            ind = np.random.choice(np.nonzero(w[:, mi])[0], size=1)[0]
            # print(np.nonzero(w[:, i]))
            # print(mi, ind)
            u = 0
            while True:
                u += 1
                ni = np.random.choice(list(set(range(N)) - set(non_zeros_sor[:(i+1)])))
                # print(ni)
                if non_zeros[ni] < mean_col:
                    # print('ter')
                    if w[ind, ni] == 0:
                        w[ind, ni] = 1
                        w[ind, mi] = 0
                        break
                if u > 100:
                    break
        # print(np.count_nonzero(w[:, mi]))

    # print(np.count_nonzero(w, axis=0))

    return w


N_y = 40
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
    X[i, inds] = 1

a_w = 7
w2 = generate_random_matrix(N_y, N_x, a_w)
# w_pair = generate_random_matrix(N_y **2, N_x, a_w)
a_y_range = np.arange(1, N_y, 2)
iters = 5

mut_info5x = np.zeros((iters, a_y_range.size))
mut_info5xp = np.zeros((iters, a_y_range.size))

error = np.zeros((a_y_range.size, X_size))

x_non_zero = 7
for iter in range(iters):
    print('New iter')
    print(iter)
    # w_fix = generate_random_matrix2(N_y, N_x, a_w)
    # print(np.std(np.count_nonzero(w_fix, axis=0)))

    w_p = generate_random_matrix(N_y, N_x, a_w)
    w_p2 = generate_random_matrix(N_x, N_y, a_w)
    w_r = np.random.binomial(1, float(a_w)/N_x, size=(N_y, N_x))
    w_r2 = np.random.binomial(1, float(a_w)/N_x, size=(N_x, N_y))
    for i, ai in enumerate(a_y_range):
        print(ai)
        Y = np.zeros((X_size, N_y), dtype=int)
        Y_precise = np.zeros((X_size, N_y), dtype=int)
        X_r = np.zeros((X_size, N_x), dtype=int)
        X_rpres = np.zeros((X_size, N_x), dtype=int)
        for k, x in enumerate(X):
            # Y[k] = kWTA2(w_fix @ x, ai).astype(int)
            # X_r[k] = kWTA2(w_fix.T @ Y[k], x_non_zero).astype(int)

            Y[k] = kWTA2(w_r @ x, ai).astype(int)
            X_r[k] = kWTA2(w_r2 @ Y[k], x_non_zero).astype(int)
            Y_precise[k] = kWTA2(w_p @ x, ai).astype(int)
            X_rpres[k] = kWTA2(w_p2 @ Y_precise[k], x_non_zero).astype(int)

        uni_x, counts_x = np.unique(X_r, return_counts=True, axis=0)
        mut_info5x[iter, i] = 1 - np.sum(counts_x * np.log2(counts_x)) / (X_size * N_x)

        uni_xp, counts_xp = np.unique(X_rpres, return_counts=True, axis=0)
        mut_info5xp[iter, i] = 1 - np.sum(counts_xp * np.log2(counts_xp)) / (X_size * N_x)


plt.plot(a_y_range / N_y,  np.mean(mut_info5x, axis=0), '-d',  markersize=10, label='kWTA')
plt.plot(a_y_range / N_y,  np.mean(mut_info5xp, axis=0), '-',  markersize=10, label='kWTA special')

plt.ylim([0, 1])
plt.xlim([0, 1])
plt.xlabel(r'$s_y$')
plt.ylabel(r'Scaled mutual information')
plt.legend()
plt.savefig('figures/mutual_information_special', bbox_inches='tight')
plt.show()
