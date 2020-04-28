# Calculate min_error vs a_w  thresh
#Paper version

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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

np.random.seed(0)
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

def get_theta(v_r, x, t_max):
    error = np.zeros(t_max)
    for tx in range(t_max):
        error[tx] = np.sum(np.abs((v_r >= tx).astype(int) - x))
    return np.argmin(error)




def get_min_error_kwta(x, w):
    a_x = np.count_nonzero(x)
    a_y_range = np.arange(5, N_y - 5, 1)
    error_kwta = np.zeros(a_y_range.size)
    a_x_range = np.arange(np.max([a_x - 10, 0]), np.min([a_x + 10, N_x]), 1)
    for i, ai in enumerate(a_y_range):
        y = kWTA2(w @ x, ai)
        error_kwta2 = np.zeros(a_x_range.size)
        for j, axi in enumerate(a_x_range):
            x_r = kWTA2(w.T @ y, axi)
            error_kwta2[j] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
        a_x_optimal = a_x_range[np.argmin(error_kwta2)]
        x_r = kWTA2(w.T @ y, a_x_optimal)
        # x_r = kWTA2(w.T @ y, a_x)
        error_kwta[i] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
    # print(error_kwta)
    return a_y_range[np.argmin(error_kwta)],  np.min(error_kwta)


def get_min_error_theta(x, w):
    a_x = np.count_nonzero(x)
    a_w = np.count_nonzero(w[0])
    theta_range = np.arange(1, np.min([a_x, a_w]) + 1)[::-1]
    a_y_range = np.arange(theta_range.size)
    error = np.zeros(theta_range.size)
    for i, ti in enumerate(theta_range):
        y = (w @ x >= ti).astype(int)
        a_y_range[i] = np.count_nonzero(y)
        if a_y_range[i]:
            z = w.T @ y
            t_x = get_theta(z, x, a_y_range[i])
            x_r = (z >= t_x).astype(int)
            error[i] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
        else:
            error[i] = N_x
    return a_y_range[np.argmin(error)], np.min(error)


def get_min_omp(x, w):
    steps = w.shape[0]
    y = np.zeros(w.shape[0])
    r = np.copy(x)
    a_x = np.count_nonzero(x)
    error = np.zeros(steps)
    a_x_range = np.arange(np.max([a_x - 10, 0]), np.min([a_x + 10, N_x]), 1)
    # print(a_x_range)
    for k in range(steps):
        z = w @ r
        z[np.nonzero(y)[0]] = -100
        ind = np.argmax(z)
        y[ind] = 1
        error_kwta2 = np.zeros(a_x_range.size)
        for j, axi in enumerate(a_x_range):
            x_r = kWTA2(w.T @ y, axi)
            error_kwta2[j] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
        a_x_optimal = a_x_range[np.argmin(error_kwta2)]
        x_r = kWTA2(w.T @ y, a_x_optimal)
        # r = 5 * x - x_r
        r = 2 * x - x_r
        # r =  x - x_r
        # error[k] = np.sum(np.abs(r))
        error[k] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
    return np.argmin(error) + 1, np.min(error)



N_x = 50
N_y = 150

a_x = 20


a_w_range = np.arange(1, N_x - 1, 1)

iters = 50
ay_kwta = np.zeros((iters, a_w_range.size))
error_kwta = np.zeros((iters, a_w_range.size))

ay_thresh = np.zeros((iters, a_w_range.size))
error_thresh = np.zeros((iters, a_w_range.size))

ay_omp = np.zeros((iters, a_w_range.size))
error_omp = np.zeros((iters, a_w_range.size))

for i in range(iters):
    print(i)
    x = generate_random_vector(N_x, a_x)
    for k, awi in enumerate(a_w_range):
        w = generate_random_matrix(N_y, N_x, awi)
        ay_thresh[i, k], error_thresh[i, k] = get_min_error_theta(x, w)

# print(np.mean(ay_kwta, axis=0))
# print(np.mean(error_kwta, axis=0))


print(np.mean(ay_thresh, axis=0))
print(np.mean(error_thresh, axis=0))



plt.errorbar(a_w_range, np.mean(error_thresh, axis=0) / N_x, np.std(error_thresh, axis=0) / N_x,
             ecolor='#ff7f0e', elinewidth=1.5, fmt='k--D', label='error')
plt.errorbar(a_w_range, np.mean(ay_thresh, axis=0) / N_y, np.std(ay_thresh, axis=0) / N_y,
            ecolor='#1f77b4', elinewidth=1.5, fmt='k-o', label=r'$s_y$')

plt.legend()
plt.xlabel(r'$a_w$')
plt.ylabel(r'Error and sparsity')
plt.xlim([0, N_x])
plt.ylim([0, 1])
plt.savefig('figures/error&ay_vs_aw_theta', bbox_inches='tight')
plt.show()

