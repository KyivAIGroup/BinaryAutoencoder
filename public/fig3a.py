# Get error vs a_y for thresholding kwta bmp
# paper version

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

def get_theta(v_r, x, t_max):
    # returns the optimal threshold of the output layer
    error_th = np.zeros(t_max)
    for tx in range(t_max):
        error_th[tx] = np.sum(np.abs((v_r >= tx).astype(int) - x))
    return np.argmin(error_th)


def get_error_theta(x, w, theta_range):
    a_x = np.count_nonzero(x)
    # a_w = np.count_nonzero(w[0])

    # theta_range = np.arange(1, np.min([a_x, a_w]) + 1)[::-1]
    a_y_range = np.zeros(theta_range.size, dtype=int)
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
    # print(a_y_range.size)
    return a_y_range, error


def get_error_kwta(x, w, a_y_range):
    a_x = np.count_nonzero(x)
    a_x_step = 5
    a_x_range = np.arange(a_x - a_x_step, a_x + a_x_step, 1)
    error_kwta = np.zeros(a_y_range.size)
    overlap = w @ x
    winners = np.argsort(overlap)
    for i, ai in enumerate(a_y_range):
        y = kWTA(overlap, ai, winners)
        # x_r2 = kWTA(w.T @ y, a_x)
        error_kwta2 = np.zeros(a_x_range.size)
        for j, axi in enumerate(a_x_range):
            x_r = kWTA(w.T @ y, axi)
            error_kwta2[j] = np.sum(np.abs(x - x_r))
        a_x_optimal = a_x_range[np.argmin(error_kwta2)]
        x_r2 = kWTA(w.T @ y, a_x_optimal)
        error_kwta[i] = np.sum(np.abs(x - x_r2))
    return error_kwta

def get_omp_opt(x, w):
    a_x = np.count_nonzero(x)
    steps = w.shape[0]
    y = np.zeros(w.shape[0])
    r = np.copy(x)
    error = np.zeros(steps)
    a_x_step = 10
    a_x_range = np.arange(a_x - a_x_step, a_x + a_x_step, 1)
    for k in range(steps):
        z = w @ r
        z[np.nonzero(y)[0]] = -100
        ind = np.argmax(z)
        y[ind] = 1
        error_kwta2 = np.zeros(a_x_range.size)
        for j, axi in enumerate(a_x_range):
            x_r = kWTA(w.T @ y, axi)
            error_kwta2[j] = np.sum(np.abs(x - x_r))
        a_x_optimal = a_x_range[np.argmin(error_kwta2)]
        # a_x_optimal = a_x
        x_r = kWTA(w.T @ y, a_x_optimal)
        r = 2 * x - x_r
        error[k] = np.sum(np.abs(x - x_r))
    return np.arange(steps), error


N_x = 50
N_y = 150

a_x = 25
a_w = 7


iters = 2000



a_y_range_omp = np.arange(N_y)
error_omp = np.zeros((iters, N_y))

theta_range = np.arange(1, np.min([a_x, a_w]) + 5)
a_y_range_thr = np.zeros((iters, theta_range.size))
error_threshold = np.zeros((iters, theta_range.size))

N_y_range = np.array([200, 500, 1000, 2000])
data = np.zeros((N_y_range.size))
import time
t = time.time()
# w = generate_random_matrix(N_y, N_x, a_w)

# plt.hist(np.count_nonzero(w, axis=0))
# plt.show()
# quit()
for j, ny in enumerate(N_y_range):
    print(ny)
    a_y_range = np.arange(5, ny - 5, int(ny * 0.01))
    error_kwta = np.zeros((iters, a_y_range.size))
    w = np.random.binomial(1, float(a_w) / N_x, size=(ny, N_x))
    for i in range(iters):
        # print(i, end=',')
        x = generate_random_vector(N_x, a_x)
        # x = np.random.binomial(1, float(a_x) / N_x, size=N_x)
        error_kwta[i] = get_error_kwta(x, w, a_y_range)
    data[j] = a_y_range[np.argmin(np.mean(error_kwta, axis=0))] / ny
    # print()
    plt.plot(a_y_range / ny, np.mean(error_kwta, axis=0) / N_x, linestyle='-', label=f'kWTA {ny}')

print(data)


print('Time: ', time.time() - t)

plt.xlabel(r'$s_y$')
plt.ylabel(r'Error')
plt.xlim([0, 1])
plt.ylim([0, 0.4])
plt.legend()
plt.savefig('figures/error_noise', bbox_inches='tight')
plt.show()
