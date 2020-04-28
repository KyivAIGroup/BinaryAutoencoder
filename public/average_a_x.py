# mean ax_kwta is close to a_x
# (it is a bit lower because it selects the lowest a_x when the error is minimal)

# for the moderate a_x a_w indeed a_x_optimal reconstruction on average at a_x (with small deviations)
# as a_x goes to extremes, deviations decreases and the optimal a_x the same as input a_x


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import norm, binom

import matplotlib as mpl
from mpl_toolkits import mplot3d
from matplotlib import cm


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



N_x = 100
N_y = 500

a_x = 40
a_w = 50

N = N_x
M = N_y


def get_min_error_kwta(x, w):
    # a_y_range = np.arange(5, N_y - 5, 1)
    a_y_range = np.arange(50, 200, 1)
    a_x_range = np.arange(a_x - 5, a_x + 10, 1)
    error_kwta = np.zeros((a_y_range.size, a_x_range.size))
    for i, ai in enumerate(a_y_range):
        y = kWTA2(w @ x, ai)
        for j, axi in enumerate(a_x_range):
            x_r = kWTA2(w.T @ y, axi)
            # print()
            error_kwta[i, j] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
    ind_y, ind_x = np.unravel_index(np.argmin(error_kwta, axis=None), error_kwta.shape)
    return a_y_range[ind_y], a_x_range[ind_x], np.min(error_kwta)

x = generate_random_vector(N, a_x)
w = generate_random_matrix(M, N, a_w)
get_min_error_kwta(x, w)



iters = 20
ay_kwta = np.zeros(iters)
ax_kwta = np.zeros(iters)
error_th = np.zeros(iters)
error_kwta = np.zeros(iters)

for i in range(iters):
    print(i)
    x = generate_random_vector(N, a_x)
    w = generate_random_matrix(M, N, a_w)
    ay_kwta[i], ax_kwta[i],  error_kwta[i] = get_min_error_kwta(x, w)


# print(ay_kwta)
print(ax_kwta)
# print(error_kwta)

# print(np.mean(ay_kwta))
print(np.mean(ax_kwta))
# print(np.mean(error_kwta))


