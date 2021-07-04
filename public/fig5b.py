# Test accuracy of ass memory for different sprasity of Y layer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import matplotlib as mpl
from scipy.special import comb

mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

mpl.rcParams['figure.figsize'] = [10.0, 8.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 800
mpl.rcParams['savefig.format'] = 'pdf'

mpl.rcParams['font.size'] = 22
mpl.rcParams['legend.fontsize'] = 22
mpl.rcParams['figure.titlesize'] = 14

########################
def kWTA(cells, k, argsorted=None):
    # n_active = max(int(sparsity * cells.size), 1)
    if argsorted is not None:
        winners = argsorted[-k:]
    else:
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


N_x = 50
N_z = 21
N_y = 2000

a_x = int(N_x * 0.5)
a_z = int(N_z * 0.5)
# print(a_z)
a_w = 7



import time
t = time.time()


def plot_curve(R, noise=0.0):
    iters = 5
    N_split = 50
    a_y_range = np.hstack([np.arange(1, N_split, 1), np.arange(N_split, N_y, int(N_y*0.01))])
    data = np.zeros((iters, a_y_range.size))
    data2 = np.zeros((iters, a_y_range.size))
    for it in range(iters):
        print(it)
        D_x = np.zeros((R, N_x), dtype=int)
        D_z = np.zeros((R, N_z), dtype=int)
        for i in range(R):
            inds = np.random.choice(N_x, size=a_x, replace=False)
            inds2 = np.random.choice(N_z, size=a_z, replace=False)
            D_x[i, inds] = 1
            D_z[i, inds2] = 1

        for k, ayi in enumerate(a_y_range):
            # print(ayi)
            D_y = np.zeros((R, N_y), dtype=int)
            w = generate_random_matrix(N_y, N_x, a_w)
            w_yz = np.zeros((N_z, N_y))
            for i in range(R):
                D_y[i] = kWTA(w @ D_x[i], ayi)
                w_yz += np.outer(D_z[i], D_y[i])


            w_yz[w_yz > 0] = 1

            #Recall
            error = np.zeros(R)
            max_error = 2 * a_z * (1 - float(a_z) / N_z) # for random retrieval
            # print(max_error)
            for i in range(R):
                if noise > 0:
                    z = kWTA(w_yz @ (D_y[i] ^ np.random.binomial(1, noise, size=N_y)), a_z)
                else:
                    z = kWTA(w_yz @ D_y[i], a_z)
                error[i] = np.sum(np.abs(D_z[i] - z)) / (max_error)

            data[it, k] = np.mean(error)
    if noise > 0:
        plt.plot(a_y_range / N_y, np.mean(data, axis=0), '-', label=f'R={R} with expansion, noise')
    else:
        plt.plot(a_y_range / N_y, np.mean(data, axis=0), '-', label=f'R={R} with expansion')
    print(f'R={R}')
    print('min error', np.min(np.mean(data, axis=0)))
    print('a_min', a_y_range[np.argmin(np.mean(data, axis=0))])
    print('s_min', a_y_range[np.argmin(np.mean(data, axis=0))] / N_y)



def plot_curve_both(R):
    iters = 5
    N_split = 50
    a_y_range = np.hstack([np.arange(1, N_split, 1), np.arange(N_split, N_y, int(N_y*0.01))])
    data = np.zeros((iters, a_y_range.size))
    data2 = np.zeros((iters, a_y_range.size))
    for it in range(iters):
        print(it)
        D_x = np.zeros((R, N_x), dtype=int)
        D_z = np.zeros((R, N_z), dtype=int)
        for i in range(R):
            inds = np.random.choice(N_x, size=a_x, replace=False)
            inds2 = np.random.choice(N_z, size=a_z, replace=False)
            D_x[i, inds] = 1
            D_z[i, inds2] = 1

        for k, ayi in enumerate(a_y_range):
            # print(ayi)
            D_y = np.zeros((R, N_y), dtype=int)
            w = generate_random_matrix(N_y, N_x, a_w)
            w_yz = np.zeros((N_z, N_y))
            w_xz = np.zeros((N_z, N_x))
            for i in range(R):
                D_y[i] = kWTA(w @ D_x[i], ayi)
                w_yz += np.outer(D_z[i], D_y[i])
                w_xz += np.outer(D_z[i], D_x[i])


            w_yz[w_yz > 0] = 1
            w_xz[w_xz > 0] = 1

            #Recall
            error = np.zeros(R)
            error2 = np.zeros(R)
            max_error = 2 * a_z * (1 - float(a_z) / N_z) # for random retrieval
            # print(max_error)
            noise = 0.0
            for i in range(R):
                if noise > 0:
                    z = kWTA(w_yz @ (D_y[i] ^ np.random.binomial(1, noise, size=N_y)), a_z)
                    z2 = kWTA(w_xz @ (D_x[i] ^ np.random.binomial(1, noise, size=N_x)), a_z)
                else:
                    z = kWTA(w_yz @ D_y[i], a_z)
                    z2 = kWTA(w_xz @ D_x[i], a_z)
                error[i] = np.sum(np.abs(D_z[i] - z)) / (max_error)
                error2[i] = np.sum(np.abs(D_z[i] - z2)) / (max_error)

            data[it, k] = np.mean(error)
            data2[it, k] = np.mean(error2)
        # print(error)
        # print(np.mean(error))
    # print(D_y[:3])
    # print(data)
    # print(time.time() - t)


    plt.plot(a_y_range / N_y, np.mean(data, axis=0), '-.', label=f'R={R} with expansion')
    plt.plot(a_y_range / N_y, np.mean(data2, axis=0), '--', label=f'R={R} without hidden layer')
    print(f'R={R}')
    print('min error', np.min(np.mean(data, axis=0)))
    print('a_min', a_y_range[np.argmin( np.mean(data, axis=0))])
    print('s_min', a_y_range[np.argmin( np.mean(data, axis=0))]/N_y)


plot_curve_both(100)
plot_curve(200)
plot_curve(200, noise=0.01)

plt.legend()
plt.ylim([0, 1.1])
plt.xlim([0, 0.2])
plt.xticks(np.arange(0, 0.2, step=0.05))
plt.xlabel(r'$s_y$')
plt.ylabel(r'Scaled retrieval error')
plt.legend()
plt.savefig('figures/association_all', bbox_inches='tight')
plt.show()

