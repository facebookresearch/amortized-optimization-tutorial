#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import jax
import jax.numpy as jnp

import os
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"]})
plt.style.use('bmh')

fig, ax = plt.subplots(figsize=(2,1.3), dpi=200)

N = 1000
y = np.linspace(-2.0, 2.0, N)
z = -y**3 - 10.*y
ax.plot(y, z, color='k')

I = N // 5
y0, z0 = y[I], z[I]
ax.scatter(y0, z0, color='#5499FF', lw=1, s=50, zorder=10, marker='.')
ax.text(y0, z0-3, r'$$\hat y^0_\theta$$', color='#5499FF',
        ha='right', va='top')

lams = np.linspace(0., 12., 15)
for lam in lams:
    z_ = z + (lam/2)*(y-y0)**2
    ax.plot(y, z_, color='k', alpha=0.2)

# ax.set_title('$$f(y) + {\lambda\over 2}||y-\hat y_0||_2^2$$', size=10)

# ax.set_xlabel('$$y$$')
# ax.xaxis.set_label_coords(.5, 0.01)

fig.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fname = 'imaml.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')
