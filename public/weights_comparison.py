# Comparison of different activation functions.
# weigth binary vs real
# binary weight from bernouli
# select most similar

# cosine similiarity improves the results!


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta




# np.random.seed(0)


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

def cosine(x, x_r):
    return np.dot(x, x_r.T) / np.sqrt(np.sum(x) * np.sum(x_r))



def generate(R, N, pdf):
    X = np.zeros((R, N))
    for i in range(R):
        X[i] = (pdf > np.random.rand(N)).astype(int)
    return X


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


def get_error(a_w, a_x, N, M):
    x = generate_random_vector(N, a_x)
    iters = 100
    a_y_var = np.arange(10, M-10, 5)
    error = np.zeros((iters, a_y_var.size))

    for j in range(iters):
        print(j, end=',')
        w = np.random.binomial(1, a_w / N, size=(M, N))
        for i, ayi in enumerate(a_y_var):
            y = kWTA2(np.dot(w, x), ayi)
            x_r = kWTA2(np.dot(w.T, y), a_x)
            error[j, i] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
    return a_y_var, np.mean(error, axis=0)


def get_error_cosine(a_w, a_x, N, M):
    x = generate_random_vector(N, a_x)
    iters = 100
    a_y_var = np.arange(10, M-10, 5)
    error = np.zeros((iters, a_y_var.size))

    for j in range(iters):
        print(j, end=',')
        w = np.random.binomial(1, a_w / N, size=(M, N))
        for i, ayi in enumerate(a_y_var):
            # y = kWTA2(np.dot(w, x), ayi)
            cos_act = [cosine(w[u], x) for u in range(N_y)]
            y = kWTA2(np.array(cos_act), ayi)
            x_r = kWTA2(np.dot(w.T, y), a_x)
            error[j, i] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))
    return a_y_var, np.mean(error, axis=0)

def get_error_narrow(a_w, a_x, N, M):
    x = generate_random_vector(N, a_x)
    iters =100
    a_y_var = np.arange(10, M-10, 5)
    error = np.zeros((iters, a_y_var.size))

    for j in range(iters):
        print(j, end=',')
        w = generate_random_matrix(M, N, a_w)
        for i, ayi in enumerate(a_y_var):
            y = kWTA2(np.dot(w, x), ayi)
            x_r = kWTA2(np.dot(w.T, y), a_x)
            error[j, i] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))

    return a_y_var, np.mean(error, axis=0)

def get_error_real(a_w, a_x, N, M):
    x = generate_random_vector(N, a_x)
    iters = 100
    a_y_var = np.arange(10, M - 10, 5)
    error = np.zeros((iters, a_y_var.size))

    for j in range(iters):
        print(j, end=',')
        w = np.random.rand(M, N)
        for i, ayi in enumerate(a_y_var):
            y = kWTA2(np.dot(w, x), ayi)
            x_r = kWTA2(np.dot(w.T, y), a_x)
            error[j, i] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))

    return a_y_var, np.mean(error, axis=0)

def get_error_real_cosine(a_w, a_x, N, M):
    x = generate_random_vector(N, a_x)
    iters =100
    a_y_var = np.arange(10, M - 10, 5)
    error = np.zeros((iters, a_y_var.size))

    for j in range(iters):
        print(j, end=',')
        w = np.random.rand(M, N)
        for i, ayi in enumerate(a_y_var):
            cos_act = [cosine(w[u], x) for u in range(N_y)]
            y = kWTA2(np.array(cos_act), ayi)
            x_r = kWTA2(np.dot(w.T, y), a_x)
            error[j, i] = np.dot(x, (1 - x_r)) + np.dot(x_r, (1 - x))

    return a_y_var, np.mean(error, axis=0)

N_x = 50
N_y = 200

a_x = 20
a_w = 30

a_range, error_exp = get_error(a_w, a_x, N_x, N_y)
a_range_cos, error_exp_cos = get_error_cosine(a_w, a_x, N_x, N_y)
a_range2, error_exp_pair = get_error_real(a_w, a_x, N_x, N_y)
a_range2_cos, error_exp_pair_cos = get_error_real_cosine(a_w, a_x, N_x, N_y)
a_range3, error_exp_narrow = get_error_narrow(a_w, a_x, N_x, N_y)
plt.plot(a_range, error_exp / N_x, label="error_bernoulli_")
plt.plot(a_range_cos, error_exp_cos / N_x, label="error_bernoulli_cosine")
plt.plot(a_range2, error_exp_pair / N_x, label="error_real")
plt.plot(a_range2_cos, error_exp_pair_cos / N_x, label="error_real_cosine")
plt.plot(a_range3, error_exp_narrow / N_x, label="error_binary")
plt.legend()
plt.ylim([0, 0.5])
plt.show()

