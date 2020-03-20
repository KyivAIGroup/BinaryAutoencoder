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
mpl.rcParams['savefig.dpi'] = 200
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

    x_axis = np.linspace(-20, 300, 100000)
    pdf_0 = norm.pdf(x_axis, mean_zeros, np.sqrt(variance_zeros)) * (N_x - a_x)
    pdf_1 = norm.pdf(x_axis, mean_ones, np.sqrt(variance_ones)) * a_x
    sum_pdf = pdf_0 + pdf_1
    thresh_x_index = np.nonzero(np.cumsum(sum_pdf) * (x_axis[10] - x_axis[9])  >= N_x - a_x)[0][0]
    thresh_x = x_axis[thresh_x_index]
    plt.fill_between(x_axis[thresh_x_index:], pdf_0[thresh_x_index:],
             facecolor="none", edgecolor="k", hatch="-", rasterized=True)
    plt.fill_between(x_axis[:thresh_x_index], pdf_1[:thresh_x_index],
                      facecolor="none",edgecolor="k", hatch='|', rasterized=True)
    plt.fill_between(x_axis[thresh_x_index:], pdf_1[thresh_x_index:],
            facecolor="none", edgecolor="k", rasterized=True)
    plt.fill_between(x_axis[:thresh_x_index], pdf_0[:thresh_x_index],
             facecolor="none", edgecolor="k", rasterized=True)

    plt.plot(x_axis, pdf_0)
    plt.plot(x_axis, pdf_1)
    plt.axvline(thresh_x, color='orange')
    plt.xlim([10, 40])
    plt.ylim([0, 4])
    plt.xlabel(r'$k$, inverse  overlap')
    plt.ylabel(r'$p(v_i = k)$')
    plt.savefig('figures/two_gauss', bbox_inches='tight')
    plt.show()




N_x = 50
N_y = 200
a_x = 20
a_w = 30
a_y = 40
get_errors(a_x, N_x, a_y, N_y, a_w)



