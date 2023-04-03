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
def f(x, y):
    z = y-jnp.sin(x)-x*1.+0.1*jnp.cos(x)
    z = z**2
    z = 1./(1.+jnp.exp(-z/80.))
    return z

N = 1000
x = np.linspace(-5.0, 5.0, N)
y = np.linspace(-10.0, 10.0, N)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig, ax = plt.subplots(figsize=(2,1.7), dpi=200)
CS = ax.contourf(X, Y, Z, cmap='Purples')

ax.text(0., 1., r'$$f(y; x)$$', color='#491386',
        bbox=dict(facecolor='white', pad=0, alpha=0.9, edgecolor='none'),
        transform=ax.transAxes, ha='left', va='top')


I = np.argmin(Z, axis=0)
xstar, ystar = x, y[I]
ax.plot(xstar, ystar, color='#AA0000', lw=3)
ax.text(.92, .8, '$$y^\star(x)$$', color='#AA0000',
        transform=ax.transAxes, ha='right', va='top')

ax.set_ylabel('$$y$$', rotation=0, labelpad=0)
ax.yaxis.set_label_coords(-.07, .44)
ax.set_xlabel('$$x$$')
ax.xaxis.set_label_coords(.5, 0.01)

fig.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fname = 'opt.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')

# Regression loss
xhat, yhat= xstar.copy(), ystar.copy()
yhat = -0.5*yhat + 0.0*xhat*np.maximum(xhat, 0.) - \
  0.23*xhat*np.minimum(xhat, 0.)

fig, ax = plt.subplots(figsize=(2,1.7), dpi=200)
CS = ax.contourf(X, Y, Z, cmap='Purples')

ax.text(0., 1., r'$$f(y; x)$$', color='#491386',
        bbox=dict(facecolor='white', pad=0, alpha=0.9, edgecolor='none'),
        transform=ax.transAxes, ha='left', va='top')

I = np.argmin(Z, axis=0)
xstar, ystar = x, y[I]
ax.plot(xstar, ystar, color='#AA0000', lw=3)
ax.text(.92, .8, '$$y^\star(x)$$', color='#AA0000',
        transform=ax.transAxes, ha='right', va='top')

ax.plot(xhat, yhat, color='#5499FF', lw=3)
ax.text(0.3, .57, r'$$\hat y_\theta(x)$$', color='#5499FF',
        bbox=dict(facecolor='white', pad=0, alpha=0.6, edgecolor='none'),
        transform=ax.transAxes, ha='left', va='bottom')

n_reg = 15
pad = 35
I = np.round(np.linspace(pad, len(y) - 1 - pad, n_reg)).astype(int)
for idx in I:
    ax.plot(
        (xstar[idx], xhat[idx]), (yhat[idx], ystar[idx]),
        color='k', lw=1, solid_capstyle='round')

ax.set_ylabel('$$y$$', rotation=0, labelpad=0)
ax.yaxis.set_label_coords(-.07, .44)
ax.set_xlabel('$$x$$')
ax.xaxis.set_label_coords(.5, 0.01)
ax.set_title('Regression-Based', fontsize=12, pad=0)

fig.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fname = 'learning-reg.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')

# Objective loss
fig, ax = plt.subplots(figsize=(2,1.7), dpi=200)
CS = ax.contourf(X, Y, Z, cmap='Purples')

ax.plot(xstar, ystar, color='#AA0000', lw=3, ls='--')
ax.plot(xhat, yhat, color='#5499FF', lw=3)

I = np.round(np.linspace(pad, len(y) - 1 - pad, n_reg)).astype(int)

df = jax.grad(f, argnums=1)

for idx in I:
    x,y = jnp.array(xhat[idx]), jnp.array(yhat[idx])
    z = f(x,y)
    dz = df(x,y)
    ax.quiver(
        xhat[idx], yhat[idx], 0., -dz,
        color='k', lw=1, scale=.2, zorder=10) #, solid_capstyle='round')

ax.set_ylabel('$$y$$', rotation=0, labelpad=0)
ax.yaxis.set_label_coords(-.07, .44)
ax.set_xlabel('$$x$$')
ax.xaxis.set_label_coords(.5, 0.01)
ax.set_title('Objective-Based', fontsize=12, pad=0)

fig.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fname = 'learning-obj.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')


# RL loss
fig, ax = plt.subplots(figsize=(2,1.5), dpi=200)
CS = ax.contourf(X, Y, Z, cmap='Purples')

ax.plot(xstar, ystar, color='#AA0000', lw=3, ls='--')
ax.plot(xhat, yhat, color='#5499FF', lw=3)

np.random.seed(2)
for _ in range(20):
    p = np.linspace(0, 3., len(xhat))
    p = p*np.flip(p)
    q = 0.04*np.random.randn(len(xhat))
    q = np.cumsum(q, axis=-1)
    q = q*np.flip(q)
    pert = 0.3*(p+q)*np.random.randn()
    ax.plot(xhat, yhat+pert, color='#5499FF', lw=1, alpha=0.3)

# ax.set_xlabel('$$x$$')
ax.xaxis.set_label_coords(.5, 0.01)
# ax.set_title('RL-Based', fontsize=12, pad=0)

fig.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fname = 'learning-rl.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')


fig, ax = plt.subplots(figsize=(2,1.7), dpi=200)

x = np.linspace(-5.0, 5.0, N)
y = np.linspace(-10.0, 10.0, N)
X, Y = np.meshgrid(x, y)
int_thresh = 0.2
Y_close_to_int = np.abs(Y - np.round(Y)) < int_thresh
Y_ints = np.round(Y)
Z = np.array(f(X, Y_ints))
Z[~Y_close_to_int] = np.inf

CS = ax.contourf(X, Y, Z, cmap='Purples')


ax.text(0., 1., r'$$f(y; x)$$', color='#491386',
        bbox=dict(facecolor='white', pad=0, alpha=0.9, edgecolor='none'),
        transform=ax.transAxes, ha='left', va='top')

I = np.argmin(Z, axis=0)
xstar, ystar = x, y[I]
ax.plot(xstar, ystar, color='#AA0000', lw=2)
ax.text(.92, .8, '$$y^\star(x)$$', color='#AA0000',
        transform=ax.transAxes, ha='right', va='top')

ax.set_ylabel('$$y$$', rotation=0, labelpad=0)
ax.yaxis.set_label_coords(-.07, .44)
ax.set_xlabel('$$x$$')
ax.xaxis.set_label_coords(.5, 0.01)

fig.tight_layout()
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.set_ylim(-10+int_thresh, 10-int_thresh)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
fname = 'opt-discrete.pdf'
plt.savefig(fname, transparent=True)
os.system(f'pdfcrop {fname} {fname}')
