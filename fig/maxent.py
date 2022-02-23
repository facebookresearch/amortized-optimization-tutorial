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

phi = jnp.array([0., .7, 6.]) # Parameters to learn

@jax.jit
def compute_y(x, phi):
    # Compute values at the discretized points in the domain
    v = jnp.exp(-(phi[0]-x)**2 + phi[1]*jnp.sin(x*phi[2]))
    y = v/sum(v) # Normalize to be a proper discrete distribution
    x = normalize_domain(x, y) # Project onto a valid distribution.
    return x, y

@jax.jit
def mean(x, y):
    return sum(x*y)

@jax.jit
def std(x, y):
    return jnp.sqrt(sum((x-mean(x,y))**2) / len(x))

@jax.jit
def entr(x, y):
    return -sum(y*jnp.log(y+1e-8))

@jax.jit
def normalize_domain(x, y):
    # Normalize the domain so that the distribution has
    # zero mean and identity variance.
    return (x - mean(x,y)) / std(x, y)

@jax.jit
def loss(x, phi):
    x, y = compute_y(x, phi)
    return -entr(x, y)

dloss_dphi = jax.jit(jax.grad(loss, argnums=1))

fig, ax = plt.subplots(figsize=(2,1.3), dpi=200)

N = 1000 # Number of discretization points in the domain

# The domain of the unprojected distribution
x_unproj = jnp.linspace(-10.0, 10.0, N)

# Plot the initialization
x, y = compute_y(x_unproj, phi)
ax.plot(x, y, color='k', alpha=0.5)
print(f'entr={entr(x,y):.2f} (mean={mean(x, y):.2f} std={std(x,y):.2f})')

for t in range(20):
    # Take a gradient step with respect to the
    # parameters of the distribution
    phi -= dloss_dphi(x_unproj, phi)
    x, y = compute_y(x_unproj, phi)
    ax.plot(x, y, color='k', alpha=0.2)
    print(f'entr={entr(x,y):.2f} (mean={mean(x, y):.2f} std={std(x,y):.2f})')

fig.tight_layout()
ax.set_xlim(-.5, .5)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fname = 'gaussian.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')
