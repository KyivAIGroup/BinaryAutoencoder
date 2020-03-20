## It seems that i were wrond somewhere. Need to redo compuatational experiments

# Get min_error vs N_y for thresh

# paper varsion

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
    N_y = w.shape[0]
    a_y_range = np.arange(5, N_y - 5, 2)
    error_kwta = np.zeros(a_y_range.size)
    for i, ai in enumerate(a_y_range):
        y = kWTA2(w @ x, ai)
        x_r = kWTA2(w.T @ y, a_x)
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
    error = np.zeros(steps)
    a_x_range = np.arange(a_x - 10, a_x + 10, 1)
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
a_x = 20
a_w = 30


iters = 100
N_min = 100
N_max = 1350

N_y_range = np.arange(N_min, N_max, 50)
a_y_kwta= np.zeros((iters, N_y_range.size))
a_y_thr = np.zeros((iters, N_y_range.size))
a_y_omp = np.zeros((iters, N_y_range.size))

error_kwta = np.zeros((iters, N_y_range.size))
error_thresh = np.zeros((iters, N_y_range.size))
error_omp = np.zeros((iters, N_y_range.size))

for j, ny in enumerate(N_y_range):
    print(ny)
    w = generate_random_matrix(ny, N_x, a_w)
    for i in range(iters):
        x = generate_random_vector(N_x, a_x)
        # a_y_kwta[i, j], error_kwta[i,j] = get_min_error_kwta(x, w)
        a_y_thr[i, j], error_thresh[i, j] = get_min_error_theta(x, w)
        # a_y_omp[i, j], error_omp[i, j] = get_min_omp(x, w)

# print(error_kwta, a_y_kwta)
print(error_thresh, a_y_thr)
# print(error_omp, a_y_omp)



# plt.plot(N_y_range / N_x,  np.mean(error_thresh, axis=0) / N_x, linestyle='--', marker='o', color='k', label='error')
# plt.plot(N_y_range / N_x, np.mean(a_y_thr, axis=0)/ N_y_range, linestyle='-', marker='o', color='k', label=r'$a_y$')

plt.errorbar(N_y_range / N_x, np.mean(error_thresh, axis=0) / N_x, np.std(error_thresh, axis=0) / N_x,
             ecolor='#ff7f0e', elinewidth=1.5, label='error', fmt='k--D')
plt.errorbar(N_y_range / N_x, np.mean(a_y_thr, axis=0) / N_y_range, np.std(a_y_thr, axis=0) / N_y_range,
             ecolor='#1f77b4', elinewidth=1.5, label=r'$s_y$', fmt='k-o')


plt.xlabel(r'$\frac{N_y}{ N_x}$')
plt.ylabel(r'Error, sparsity')

plt.legend()
plt.ylim([0, 0.4])
plt.xlim([N_min/ N_x, N_max/N_x])
plt.savefig('figures/error&sy_vs_Ny_theta', bbox_inches='tight')

plt.show()

