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

phi = jnp.array([0., 1.5, .7, 6.]) # Parameters to learn

@jax.jit
def compute_dist(x, phi):
    # Compute values at the discretized points in the domain
    v = jnp.exp(-0.5*((x-phi[0])/phi[1])**2 + phi[2]*jnp.sin(x*phi[3]))
    dx = x[1:]-x[:-1]
    y = v/sum(v[1:]*dx) # Normalize to be a proper distribution.
    flow_x = flow(x, y) # Constrain the mean and variance.
    J_flow = jnp.diag(jax.jacfwd(flow)(x, y))
    flow_y = y / J_flow
    return flow_x, flow_y

@jax.jit
def mean(x, y):
    dx = x[1:]-x[:-1]
    x = x[1:]
    y = y[1:]
    return sum(x*y*dx)

@jax.jit
def std(x, y):
    mu = mean(x,y)
    dx = x[1:]-x[:-1]
    x = x[1:]
    y = y[1:]
    return jnp.sqrt(sum(((x-mu)**2)*y*dx))

@jax.jit
def entr(x, y):
    dx = x[1:]-x[:-1]
    y = y[1:]
    return -sum(y*jnp.log(y+1e-8)*dx)

@jax.jit
def flow(x, y):
    # Normalize the domain so that the distribution has
    # zero mean and identity variance.
    return (x - mean(x,y)) / std(x, y)

@jax.jit
def loss(x, phi):
    x, y = compute_dist(x, phi)
    return -entr(x, y)

dloss_dphi = jax.jit(jax.grad(loss, argnums=1))

fig, ax = plt.subplots(figsize=(2,1.3), dpi=200)

N = 1000 # Number of discretization points in the domain

# The domain of the unprojected distribution
x_unproj = jnp.linspace(-5.0, 5.0, N)

# Plot the initialization
x, y = compute_dist(x_unproj, phi)
ax.plot(x, y, color='k', alpha=0.5)
print(f'entr={entr(x,y):.2f} (mean={mean(x, y):.2f} std={std(x,y):.2f})')

for t in range(20):
    # Take a gradient step with respect to the
    # parameters of the distribution
    phi -= dloss_dphi(x_unproj, phi)
    x, y = compute_dist(x_unproj, phi)
    ax.plot(x, y, color='k', alpha=0.2)
    print(f'entr={entr(x,y):.2f} (mean={mean(x, y):.2f} std={std(x,y):.2f})')

fig.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fname = 'maxent.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')
