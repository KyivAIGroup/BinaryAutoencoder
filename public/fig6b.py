## Visualisation of error vs threshold

# for the kwta theoretical formulat for optimal threshold gives not bad results, but not ideal.
# why experimantal theta optimal is not integer (mean value)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import norm, binom

import matplotlib as mpl

import matplotlib as mpl


mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 800
mpl.rcParams['savefig.format'] = 'pdf'

mpl.rcParams['font.size'] = 18
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['figure.titlesize'] = 14

# np.random.seed(0)


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


#
# def get_theta(vec, a_x, theta_max):
#     theta_r = np.arange(0, theta_max)
#     ind_min = np.argmin([np.abs(np.count_nonzero(vec >= t) - a_x) for t in theta_r])
#     return theta_r[ind_min]


def get_theta(v_r, x, t_max):
    error_th = np.zeros(t_max)
    for tx in range(t_max):
        error_th[tx] = np.sum(np.abs((v_r >= tx).astype(int) - x))
    return np.argmin(error_th)


N_x = 50
N_y = 200
a_w = 30


def get_curve(N_x, N_y, a_w):

    iters = 100

    a_x_range = np.arange(2, N_x, 3)

    theta_optimal = np.zeros((a_x_range.size, iters))
    theta_optimal_hyp = np.zeros(a_x_range.size)
    for e, axi in enumerate(a_x_range):
        x = generate_random_vector(N_x, axi)
        print(axi)
        theta_var = np.arange(1, np.min([axi, a_w]) + 1, 1)
        a_y_range = np.zeros(theta_var.size)
        error = np.zeros((iters, theta_var.size))
        for it in range(iters):
            # if it % 10 == 0:
            #     print(it)
            w = generate_random_matrix(N_y, N_x, a_w)
            a_y_range = np.zeros(theta_var.size, dtype=int)

            for i, thi in enumerate(theta_var):
                y = (np.dot(w, x) >= thi).astype(int)
                a_y_range[i] = np.count_nonzero(y)
                if a_y_range[i]:
                    z = w.T @ y
                    t_x = get_theta(z, x, a_y_range[i])
                    x_r = (z >= t_x).astype(int)
                    error[it, i] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
                else:
                    # break
                    error[it, i] = N_x
            theta_optimal[e, it] = np.argmin(error[it])
        theta_optimal_hyp[e] = axi * a_w / N_x + 1
    return a_x_range, theta_optimal, theta_optimal_hyp


a_x_range, theta_optimal, theta_optimal_hyp = get_curve(N_x, N_y, a_w)


# plt.errorbar(a_x_range, np.mean(theta_optimal, axis=1), yerr=np.std(theta_optimal, axis=1), label='exp')
# plt.plot(a_x_range, theta_optimal_hyp, label='hyp')
# plt.legend()
# plt.show()

plt.plot(a_x_range/ N_x, np.abs(np.mean(theta_optimal, axis=1) - theta_optimal_hyp), label="50")


N_x = 150
N_y = 300
a_w = 75

a_x_range2, theta_optimal2, theta_optimal_hyp2 = get_curve(N_x, N_y, a_w)
plt.plot(a_x_range2 / N_x, np.abs(np.mean(theta_optimal2, axis=1) - theta_optimal_hyp2), label="150")


plt.xlabel(r'$s_x$, input sparsity ')
plt.ylabel(r'Threshold error')
plt.xlim([0, 1])
plt.savefig('figures/thresh_diff', bbox_inches='tight')
plt.show()
