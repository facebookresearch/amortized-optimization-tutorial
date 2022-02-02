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

# Initial problem with x^\star
N = 1000
x = np.linspace(-3.8, 5.0, N)
# y = -(x**2)*np.sin(x)
# y = (np.cos(x)**2) #- np.abs(x)
y = -x**2 + 7*np.sin(x+np.pi)
y = y/8.

nrow, ncol = 1, 2
fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*2.5,nrow*1.1), dpi=200)

ax = axs[0]
ax.axhline(0, color='k')
ax.plot(x, y-y.min(), color='k')
ustar = x[y.argmax()]
ax.axvline(ustar, color='#AA0000')
ax.text(0.13, 0.09, '$$\pi^\star(x)$$', color='#AA0000',
        transform=ax.transAxes, ha='left', va='bottom')

ax.axvline(ustar+2, color='#5499FF')
ax.text(0.54, 0.09, r'$$\pi_\theta(x)$$', color='#5499FF',
        transform=ax.transAxes, ha='left', va='bottom')

ax.arrow(x=ustar+2, y=0.5, dx=-0.5, dy=0.,
         width=0.1, color='#5499FF', zorder=10)

ax.text(0.7, 0.44, '$$Q(x, u)$$',
        transform=ax.transAxes, ha='left', va='bottom')

ax.set_xlabel('$$u$$')
ax.xaxis.set_label_coords(.5, 0.01)
ax.set_title('Deterministic Policy', fontsize=12, pad=5)
# ax.set_ylabel('$$Q(x, u)$$', rotation=0, labelpad=0)
# ax.yaxis.set_label_coords(-.1, .44)


ax = axs[1]
y = np.exp(y)
y -= y.min()
ax.plot(x, y, color='k') #, zorder=10)
ax.set_xlabel('$$u$$')
ax.xaxis.set_label_coords(.5, 0.01)

mu, sigma = ustar, 0.8
ystar = np.exp(-.5*((x-mu)/sigma)**2) #/ (sigma*np.sqrt(2.*np.pi))
ystar = ystar * y.sum() / ystar.sum()
ax.plot(x, ystar, color='#AA0000')

mu, sigma = ustar+2, 1.5
yhat = np.exp(-.5*((x-mu)/sigma)**2) #/ (sigma*np.sqrt(2.*np.pi))
yhat = yhat * y.sum() / yhat.sum()
ax.plot(x, yhat, color='#5499FF')

# I = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
I = [250, 300, 350, 400, 650, 700, 750, 800]
for i in I:
    ax.arrow(x=x[i], y=yhat[i], dx=-0.5, dy=0.,
             width=0.05,
             color='#5499FF',
             zorder=10)

ax.text(0.37, 0.74, '$$\pi^\star(x)$$', color='#AA0000',
        transform=ax.transAxes, ha='left', va='bottom')
ax.text(0.6, 0.45, r'$$\pi_\theta(x)$$', color='#5499FF',
        transform=ax.transAxes, ha='left', va='bottom')
ax.text(0., 0.43, '$$\mathcal{Q}(x, u)$$',
        transform=ax.transAxes, ha='left', va='bottom')
ax.axhline(0., color='k',zorder=-1)

# ax.set_ylabel('$${\mathcal{Q}}(x, u)$$', rotation=0, labelpad=0)
# ax.yaxis.set_label_coords(-.1, .44)

fig.tight_layout()
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

ax.set_title('Stochastic Policy', fontsize=12, pad=5)

fname = 'ctrl.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')
