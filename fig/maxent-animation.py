#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import jax
import jax.numpy as jnp

import shutil
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsfonts}",
})


# We will define a 1D density parameterized by \phi to maximize over
# with gradient steps, and implement this by discretizing over the domain.
phi = jnp.array([0., 1.5, .7, 6.])

@jax.jit
def compute_dist(x, phi):
    # Compute values of at the discretized points in the domain.
    v = jnp.exp(-0.5*((x-phi[0])/phi[1])**2 + phi[2]*jnp.sin(x*phi[3]))
    dx = x[1:]-x[:-1]
    y = v/sum(v[1:]*dx) # Normalize to be a proper distribution.
    flow_x = flow(x, y) # Constrain the mean and variance.

    # Compute the new probabilities.
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


# Prepare the output directory
d = 'maxent-animation'
if os.path.exists(d):
    shutil.rmtree(d)
os.makedirs(d)


def plot(t):
    nrow, ncol = 1, 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*2), dpi=200,
                            gridspec_kw={'wspace': .3, 'hspace': 0})
    ax = axs[0]
    ax.plot(entrs, color='k')
    ax.set_xlabel('Updates', fontsize=10)
    ax.set_title(r'Entropy ($\mathbb{H}_p[X]$)', fontsize=10)
    ax.set_xlim(0, n_step)
    ax.set_ylim(1.3, 1.45)

    ax = axs[1]
    ax.plot(x, y, color='k')
    ax.set_ylim(0, 0.7)
    ax.set_xlim(-3, 3)
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_title('$p(x)$', fontsize=10)

    for ax in axs:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    fig.suptitle(
        r'$\max_{p} \mathbb{H}_p[X]\; \rm{subject\; to}\; \mathbb{E}_p[X] = \mu\;\rm{and}\;\rm{Var}_p[X]=\Sigma$')
    fig.subplots_adjust(top=0.7)

    fname = f'{d}/{t:04d}.png'
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    # os.system(f'pdfcrop {fname} {fname}')
    os.system(f'convert -trim {fname} {fname}')


# jitted derivative of the loss with respect to phi
dloss_dphi = jax.jit(jax.grad(loss, argnums=1))

# Number of discretization points in the domain
# Decrease this to run faster
N = 1000

# The domain of the unprojected distribution
x_unproj = jnp.linspace(-5.0, 5.0, N)

entrs = []
x, y = compute_dist(x_unproj, phi)
entrs.append(entr(x,y))
print(f'entr={entr(x,y):.2f} (mean={mean(x, y):.2f} std={std(x,y):.2f})')

# The step size can be much larger but it's set to this for the animation.
n_step = 100
step_size = 0.13
for t in range(n_step):
    # Take a gradient step with respect to the
    # parameters of the distribution
    phi -= step_size*dloss_dphi(x_unproj, phi)
    x, y = compute_dist(x_unproj, phi)
    entrs.append(entr(x,y))
    print(f'entr={entr(x,y):.2f} (mean={mean(x, y):.2f} std={std(x,y):.2f})')

    plot(t)

# By the end, we see that the entropy is the true maximal entropy
# of the Gaussian of (1/2)log(2\pi)+(1/2) \approx 1.42.

os.system(f'convert -delay 10 -loop 0 {d}/*.png {d}/maxent.gif')
