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

N = 1000
x = np.linspace(-5., 5.0, N)

fig, ax = plt.subplots(figsize=(2,1.3), dpi=200)

y = x
ax.plot(x, y, color='k', linestyle='--', alpha=.5)

y = -2.*np.sin(x)+0.9*x*(1+0.1*np.cos(x))**2
ax.plot(x, y, color='k')

fp = max(x[np.abs(y-x) <= 5e-3]) # Numerically find the fixed-point :)
ax.scatter([0], [0], color='#AA0000', lw=1, s=70, zorder=10, marker='*')
ax.scatter([fp], [fp], color='#AA0000', lw=1, s=70, zorder=10, marker='*')
ax.scatter([-fp], [-fp], color='#AA0000', lw=1, s=70, zorder=10, marker='*')

# ax.set_ylabel('$$g(y)$$', rotation=0, labelpad=0)
# ax.yaxis.set_label_coords(-.07, .44)
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
fname = 'fp.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')
