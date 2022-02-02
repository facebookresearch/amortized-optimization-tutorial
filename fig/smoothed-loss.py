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

fig, ax = plt.subplots(figsize=(2.5,1.5), dpi=200)

def f(x):
    return np.cos(x) + 0.2*np.abs(x-np.pi/2)

N = 100
x = np.linspace(-4.*np.pi, 2*np.pi, N)
y = f(x)
ax.plot(x, y, color='k')

sigmas = [1., 1.5, 2.5]
for sigma in sigmas:
    ys = []
    # Inefficiently doing this...
    for xi in x:
        eps = sigma*np.random.randn(50000)
        yi = np.mean(f(xi+eps))
        ys.append(yi)
    ax.plot(x, ys, alpha=1., lw=2)

# ax.set_xlabel(r'$$\theta$$')
# ax.xaxis.set_label_coords(.5, 0.01)
# ax.set_ylabel(r'$${\mathcal L}(\hat y_\theta)$$', rotation=0, labelpad=0)
# ax.yaxis.set_label_coords(-.07, .44)
# ax.set_ylabel('$$y$$', rotation=0, labelpad=0)
# ax.xaxis.set_label_coords(.5, 0.01)

fig.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fname = 'smoothed-loss.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')
