# Similarity preservation tests
# Calculate mean average precision

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

mpl.rcParams['font.size'] = 24
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['figure.titlesize'] = 14



def kWTA(cells, sparsity):
    n_active = max(int(sparsity * cells.size), 1)
    winners = np.argsort(cells)[-n_active:]
    sdr = np.zeros(cells.shape, dtype=cells.dtype)
    sdr[winners] = 1
    return sdr


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

def generate_random_matrix_var_ax(R, N, a_x_min, a_x_max):
    matrix = np.zeros((R, N), dtype=int)
    for i in range(R):
        matrix[i] = generate_random_vector(N, np.random.randint(a_x_min, a_x_max))
    return matrix

def get_theta(v_r, x, t_max):
    error = np.zeros(t_max)
    for tx in range(t_max):
        error[tx] = np.sum(np.abs((v_r >= tx).astype(int) - x))
    return np.argmin(error)


def get_kwta(x, w, a_y):
    y = kWTA2(w @ x, a_y)
    return y

def get_omp(x, w, a_y):
    y = np.zeros(w.shape[0])
    r = np.copy(x)
    a_x = np.count_nonzero(x)
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

def hamming(x, x_r):
    return (np.dot(x.T, x_r) + np.dot((1 - x).T, (1 - x_r))) / x.size

def hamming_dist(x, x_r):
    return (np.dot(x.T, 1 - x_r) + np.dot((1 - x).T, x_r)) / x.size

def L1(x, x_r):
    return np.linalg.norm(x - x_r, 1) / x.size

def euclidian(x, x_r):
    return np.linalg.norm(x-x_r)

def cosine(x, x_r):
    # return x @ x_r.T / (np.linalg.norm(x) * np.linalg.norm(x_r))
    return x @ x_r.T / (np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(x_r**2)))

def get_theta(x, w, ti):
    return (w @ x >= ti).astype(int)

def get_overlap_theta(w, ti, num_closest):
    num_select = 50
    inds_to_select = np.random.choice(num_load, num_select, replace=False)
    # X_train = X[:num_select]
    X_train = X[inds_to_select]
    inds_closest = np.zeros((num_select, num_load))
    for i, x in enumerate(X_train):
        inds_closest[i] = np.argsort(np.sum((X - x) ** 2, axis=1)).astype(int)

    # theta_range = np.arange(1, np.min([a_x, a_w]) + 1)[::-1]
    Y = np.zeros((num_load, N_y))
    for i, xi in enumerate(X):
        Y[i] = get_theta(xi, w, ti)

    Y_train = Y[inds_to_select]

    inds_closest_y = np.zeros((num_select, num_load))
    for i, y in enumerate(Y_train):
        inds_closest_y[i] = np.argsort(np.sum(np.abs(Y - y), axis=1))
    map = np.zeros(num_select)
    for u in range(num_select):
        inds_true = inds_closest[u][:num_closest]
        inds_pred = inds_closest_y[u][:num_closest]
        prec = np.zeros(num_closest)
        s = 1
        for k, xi in enumerate(inds_pred):
            if xi in inds_true:
                prec[k] = s
                s += 1

        map[u] = np.mean(prec / np.arange(1, num_closest + 1))
    a_y = np.mean(np.count_nonzero(Y, axis=1))
    return a_y, np.mean(map)




def get_overlap(w, a_y, num_closest):
    num_select = 100
    inds_to_select = np.random.choice(num_load, num_select, replace=False)
    # X_train = X[:num_select]
    X_train = X[inds_to_select]
    inds_closest = np.zeros((num_select, num_load))
    for i, x in enumerate(X_train):
        inds_closest[i] = np.argsort(np.sum((X - x) ** 2, axis=1)).astype(int)
        # inds_closest[i] = np.argsort([cosine(X[j], x) for j in range(num_load)])[::-1]
    # print('done input')
    Y = np.zeros((num_load, N_y))
    Y_omp = np.zeros((num_load, N_y))
    for i, xi in enumerate(X):
        Y[i] = get_kwta(xi, w, a_y)
        Y_omp[i] = get_omp(xi, w, a_y)

    # print('done kwta')
    # Y_train = Y[:num_select]
    Y_train = Y[inds_to_select]

    inds_closest_y = np.zeros((num_select, num_load))
    inds_closest_y_omp = np.zeros((num_select, num_load))
    for i, y in enumerate(Y_train):
        inds_closest_y[i] = np.argsort(np.sum(np.abs(Y - y), axis=1))
        inds_closest_y_omp[i] = np.argsort(np.sum(np.abs(Y_omp - y), axis=1))
        # inds_closest_y[i] = np.argsort([cosine(Y[j], y) for j in range(num_load)])[::-1]
        # inds_closest_y_omp[i] = np.argsort([cosine(Y_omp[j], y) for j in range(num_load)])[::-1]
    # print('done output')
    overlap = np.zeros(num_select)

    map = np.zeros(num_select)
    map_omp = np.zeros(num_select)
    for u in range(num_select):
        inds_true = inds_closest[u][:num_closest]
        inds_pred = inds_closest_y[u][:num_closest]
        inds_pred_omp = inds_closest_y_omp[u][:num_closest]
        prec = np.zeros(num_closest)
        prec_omp = np.zeros(num_closest)
        s = 1
        for k, xi in enumerate(inds_pred):
            if xi in inds_true:
                prec[k] = s
                s += 1
        s = 1
        for k, xi in enumerate(inds_pred_omp):
            if xi in inds_true:
                prec_omp[k] = s
                s += 1
        map[u] = np.mean(prec / np.arange(1, num_closest+1))
        map_omp[u] = np.mean(prec_omp / np.arange(1, num_closest+1))
    # print('done overlap')
    return np.mean(map), np.mean(map_omp)


num_load = 1000
N_x = 50
N_y = 200
a_w = 30
a_x = 20

X = generate_random_matrix(num_load, N_x, a_x)
# X = generate_random_matrix_var_ax(num_load, N_x, 10, 40)

a_y_range = np.arange(1, N_y, 9)
# a_y_range = np.arange(100, 101, 100)

num_closest = 20

iters = 20
overlap = np.zeros((iters, a_y_range.size))
overlap_omp = np.zeros((iters, a_y_range.size))
import time
t = time.process_time()


theta_range = np.arange(1, np.min([a_x, a_w]) + 1)[::-1]
a_y_theta = np.zeros((iters, theta_range.size))
overlap_theta = np.zeros((iters, theta_range.size))
for i in range(iters):
    w = generate_random_matrix(N_y, N_x, a_w)
    # w = generate_random_matrix_var_ax(N_y, N_x, a_w)
    # w = np.random.rand(N_y, N_x)
    print(i)
    for k, ti in enumerate(theta_range):
        a_y_theta[i, k], overlap_theta[i, k] = get_overlap_theta(w, ti, num_closest)
        print(ti, a_y_theta[i, k])
    for j, ai in enumerate(a_y_range):
        print(ai)
        overlap[i, j], overlap_omp[i, j] = get_overlap(w, ai, num_closest)

print(a_y_theta)
# print(overlap)
print(np.mean(overlap, axis=0))
print(np.mean(overlap_omp, axis=0))
# print(np.std(overlap, axis=0))

elapsed_time = time.process_time() - t
print('Time: ', elapsed_time)
plt.plot(a_y_range / N_y, np.mean(overlap, axis=0), '-d', label='kWTA')
plt.plot(a_y_range/ N_y, np.mean(overlap_omp, axis=0), '-v', label='BMP')
plt.plot(np.mean(a_y_theta, axis=0) / N_y, np.mean(overlap_theta, axis=0), '--o', label='Threshold')
plt.legend()
# plt.ylim([0, 0.5])
plt.xlabel(r'$s_y$')
plt.ylabel(r'Similarity, mAP')
plt.savefig('figures/similarity_all_sync', bbox_inches='tight')
plt.show()

