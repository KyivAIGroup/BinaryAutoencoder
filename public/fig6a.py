## Figure graph overlap

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
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
mpl.rcParams['figure.titlesize'] = 20

N_x = 50
N_y = 200
a_x = 20
a_w = 30
threshold = 14

overlap_range = np.arange(np.min([a_x, a_w]))
overlap = np.zeros(np.min([a_x, a_w]))

for i, k in enumerate(overlap_range):
    overlap[i] = comb(a_x, k) * comb(N_x - a_x, a_w - k) / comb(N_x, a_w)

plt.bar(overlap_range[:threshold] + 0.5, overlap[:threshold], width=1, color='white', edgecolor='black' )
plt.bar(overlap_range[threshold:] + 0.5, overlap[threshold:], width=1, color='grey', edgecolor='black' )
plt.axvline(threshold, color='orange')
plt.xticks(overlap_range)
plt.ylabel('p(z=k), probability')
plt.xlabel('z, overlap')
plt.xlim([5, 20])
plt.savefig('figures/overlap')
plt.show()


