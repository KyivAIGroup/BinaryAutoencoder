## It seems that i were wrond somewhere. Need to redo compuatational experiments

# Get error vs a_y for thresholding
# Paper version

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
    error_th = np.zeros(t_max)
    for tx in range(t_max):
        error_th[tx] = np.sum(np.abs((v_r >= tx).astype(int) - x))
    return np.argmin(error_th)


def get_error_theta(x, w):
    a_x = np.count_nonzero(x)
    a_w = np.count_nonzero(w[0])
    theta_range = np.arange(1, np.min([a_x, a_w]) + 1)[::-1]
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
            # break
            error[i] = N_x

    return a_y_range, error


def get_error_kwta(x, w):
    a_y_range = np.arange(5, N_y - 5, 1)
    a_x_range = np.arange(a_x - 10, a_x + 10, 1)
    # error_kwta = np.zeros((a_y_range.size, a_x_range.size))
    error_kwta = np.zeros(a_y_range.size)
    for i, ai in enumerate(a_y_range):
        y = kWTA2(w @ x, ai)
        x_r = kWTA2(w.T @ y, a_x)
        error_kwta2 = np.zeros(a_x_range.size)
        for j, axi in enumerate(a_x_range):
            x_r = kWTA2(w.T @ y, axi)
            error_kwta2[j] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
        a_x_optimal = a_x_range[np.argmin(error_kwta2)]
        # print(a_x_optimal)
        x_r2 = kWTA2(w.T @ y, a_x_optimal)
        error_kwta[i] = np.dot(x, (1 - x_r2)) + np.dot(x_r2, (1 - x))
    # print(error_kwta)
    return a_y_range,  error_kwta

def get_omp_opt(x, w):
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
    return np.arange(steps), error


N_x = 50
N_y = 150

a_x = 20
a_w = 30


iters = 100
a_y_range = np.arange(5, N_y - 5, 1)
theta_range = np.arange(1, np.min([a_x, a_w]) + 1)

a_y_range_dt = np.zeros((iters, a_y_range.size))
error_kwta = np.zeros((iters, a_y_range.size))

a_y_range_omp = np.arange(N_y)
# a_y_range_omp = np.zeros((iters, N_y))
error_omp = np.zeros((iters, N_y))

a_y_range_thr = np.zeros((iters, theta_range.size))
error_threshold = np.zeros((iters, theta_range.size))

x = generate_random_vector(N_x, a_x)
for i in range(iters):
    w = generate_random_matrix(N_y, N_x, a_w)
    print(i)
    # a_y_range_dt[i], error_kwta[i] = get_error_kwta(x, w)
    a_y_range_thr[i], error_threshold[i] = get_error_theta(x, w)
    # _, error_omp[i] = get_omp_opt(x, w)
    # plt.plot( a_y_range_thr[i]/ N_y, error_threshold[i]/ N_x)


print(np.mean(error_threshold, axis=0))
print(np.std(error_threshold, axis=0))

print(np.mean(a_y_range_thr, axis=0))
print(np.std(a_y_range_thr, axis=0))

# quit()
# plt.plot(a_y_range / N_y, np.mean(error_kwta, axis=0) / N_x, label='kwta')
# plt.plot(a_y_range_omp / N_y, np.mean(error_omp, axis=0) / N_x, label='omp')
# plt.plot(np.mean(a_y_range_thr, axis=0) / N_y, np.mean(error_threshold, axis=0) / N_x, linestyle='-', marker='o', color='k', label='threshold')
plt.errorbar(np.mean(a_y_range_thr, axis=0)/ N_y, np.mean(error_threshold, axis=0)/ N_x,
             yerr=np.std(error_threshold, axis=0)/ N_x, xerr=np.std(a_y_range_thr, axis=0)/ N_y,
             ecolor='#1f77b4', elinewidth=1.5, label='threshold', fmt='k-o')
plt.xlabel(r'$s_y$')
plt.ylabel('Error')
plt.xlim([0, 1])
plt.ylim([0, 1])
# plt.legend()
plt.savefig('figures/error_vs_ay_theta', bbox_inches='tight')
plt.show()

