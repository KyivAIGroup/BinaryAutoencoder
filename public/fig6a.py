## Here I will test the math theory.
# Slightly changed task. Given x and w are random bernuli. The overlap fuunction will be simplified.
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import norm, binom

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

# Overlap
def intersection(k, a_x, a_w, N):
    return comb(a_x, k) * comb(N - a_x, a_w - k) / comb(N, a_w)


def get_errors(a_x, N_x, a_y, N_y, a_w):
    max_overlap = np.min([a_x, a_w])
    overlap_range = np.arange(max_overlap + 1)
    overlap = np.zeros(max_overlap + 1)
    for i, k in enumerate(overlap_range):
        overlap[i] = comb(a_x, k) * comb(N_x - a_x, a_w - k) / comb(N_x, a_w) * N_y
    threshold = np.nonzero(np.cumsum(overlap) > N_y - a_y)[0][0]
    # max_overlap = np.where(overlap > 0.0001)[0][-1]  # this is to speed up the following computations
    # plt.plot(overlap_range, overlap)
    # plt.axvline(threshold, color='red')
    # plt.show()
    # quit()
    mean_range = np.arange(threshold, max_overlap+1)

    residual = ((N_y - a_y) - np.cumsum(overlap)[threshold - 1])
    # residual = 0

    mean_ones = 0
    variance_ones = 0
    mean_zeros = 0
    variance_zeros = 0
    for i, o in enumerate(mean_range):
        if o == threshold:
            r = residual
        else:
            r = 0
        mean_ones += float(o) / a_x * (overlap[o] - r)
        variance_ones += float(o) / a_x * (1 - float(o) / a_x) * (overlap[o] - r)
        mean_zeros += float(a_w - o) / (N_x - a_x) * ((overlap[o] - r))
        variance_zeros += float(a_w - o) / (N_x - a_x) * (1 - float(a_w - o) / (N_x - a_x)) * ((overlap[o] - r))

    cdf_axis = np.linspace(0, 200, 1000000)
    cdf_0 = norm.cdf(cdf_axis, mean_zeros, np.sqrt(variance_zeros))
    cdf_1 = norm.cdf(cdf_axis, mean_ones, np.sqrt(variance_ones))
    theta_custom_index = np.nonzero(cdf_0*(N_x - a_x) + cdf_1 * a_x >= N_x - a_x)[0][0]
    false_positive = (1 - cdf_0[theta_custom_index]) * (N_x - a_x)
    false_negative = cdf_1[theta_custom_index] * a_x
    print(false_positive, false_negative)
    return false_positive, false_negative


# N_x = 200
# N_y = 500
# a_x = 50
# a_w = 70
#

N_x = 50
N_y = 200
a_x = 20
a_w = 30

# fp, fn = get_errors(a_x, N_x, 100, N_y, a_w)

# quit()
a_y_var = np.arange(2, N_y, 5)
error = np.zeros(a_y_var.shape)
s_y_range = np.zeros(a_y_var.shape)
difference = 0
for i, ayi in enumerate(a_y_var):
    print(i, ayi)
    fp, fn = get_errors(a_x, N_x, ayi, N_y, a_w)
    difference += np.abs(fp - fn)
    error[i] = (fp + fn)

print(difference)

########################
###### Experiment ######
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


def get_error(a_w, a_x):
    x = generate_random_vector(N_x, a_x)
    iters = 50
    # a_y_var = np.arange(2, N_y-1, 5)
    error = np.zeros((iters, a_y_var.size))
    for j in range(iters):
        # print(j, end=',')
        w = generate_random_matrix(N_y, N_x, a_w)
        for i, ayi in enumerate(a_y_var):
            y = kWTA2(np.dot(w, x), ayi)
            x_r = kWTA2(np.dot(w.T, y), a_x)
            error[j, i] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
    return np.mean(error, axis=0) #, np.std(error, axis=0)

iters_dt = 20
data = np.zeros((iters_dt, a_y_var.size))
for i in range(iters_dt):
    print(i)
    data[i] = get_error(a_w, a_x)

plt.plot(a_y_var / N_y, (np.mean(data, axis=0)) / N_x, label='error exp')
plt.plot(a_y_var/ N_y, error / N_x, label="error_theory")
plt.legend()
plt.ylim([0., 0.5])
# plt.savefig('figures/error')
plt.show()


plt.errorbar(a_y_var / N_y, (np.mean(data, axis=0) - error)/N_x,
             yerr= np.std(data, axis=0) / N_x,
             ecolor='#1f77b4', elinewidth=1.5, label='threshold', fmt='k-o')
plt.xlabel(r'$s_y$, hidden layer sparsity')
plt.ylabel(r'Average errors difference')
plt.xlim([0, 1])
plt.savefig('figures/error_diff', bbox_inches='tight')
plt.show()

