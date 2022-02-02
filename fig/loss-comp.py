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

fig, ax = plt.subplots(figsize=(1.5,1.5), dpi=200)

N = 1000
x = np.linspace(-5.0, 5.0, N)
y = np.linspace(-5.0, 5.0, N)
X, Y = np.meshgrid(x, y)
a,b = 0., 10.
Z = X**2 + Y**2 + 1.4*X*Y
Z = 1./(1.+np.exp(-Z/10.))

fig, ax = plt.subplots(figsize=(2,1.7), dpi=200)
CS = ax.contourf(X, Y, Z, cmap='Purples', alpha=0.8)

Z = X**2 + Y**2
CS = ax.contour(X, Y, Z, colors='k', alpha=.7, linewidths=1, levels=5)

ax.scatter([0], [0], color='#AA0000', lw=1, s=50, zorder=10, marker='*')

ax.set_ylabel('$$y_1$$', rotation=0, labelpad=0)
ax.yaxis.set_label_coords(-.07, .44)
ax.set_xlabel('$$y_0$$')
ax.xaxis.set_label_coords(.5, 0.01)

ax.text(0., 1., r'$$f(y; x)$$', color='#491386',
        bbox=dict(facecolor='white', pad=0, alpha=0.9, edgecolor='none'),
        transform=ax.transAxes, ha='left', va='top')

ax.text(.3, .3, '$$y^\star(x)$$', color='#AA0000',
        ha='left', va='bottom')

fig.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fname = 'loss-comp.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')
