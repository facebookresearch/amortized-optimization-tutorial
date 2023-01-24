#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch import nn
import numpy as np
import os

import matplotlib.pyplot as plt
plt.style.use('bmh')

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)

def celestial_to_euclidean(ra, dec):
    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)
    return x, y, z

def euclidean_to_celestial(x, y, z):
    sindec = z
    cosdec = np.sqrt(x*x + y*y)
    sinra = y / cosdec
    cosra = x / cosdec
    ra = np.arctan2(sinra, cosra)
    dec = np.arctan2(sindec, cosdec)
    return ra, dec

def euclidean_to_celestial_th(x, y, z):
    sindec = z
    cosdec = (x*x + y*y).sqrt()
    sinra = y / cosdec
    cosra = x / cosdec
    ra = torch.atan2(sinra, cosra)
    dec = torch.atan2(sindec, cosdec)
    return ra, dec


def sphere_dist_th(x,y):
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    assert x.ndim == y.ndim == 2
    inner = (x*y).sum(-1)
    return torch.arccos(inner)

class c_convex(nn.Module):
    def __init__(self, n_components=4, gamma=0.5, seed=None):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma

        # Sample a random c-convex function
        if seed is not None:
            torch.manual_seed(seed)
        self.ys = torch.randn(n_components, 3)
        self.ys = self.ys / torch.norm(self.ys, 2, dim=-1, keepdim=True)
        self.alphas = .7*torch.rand(self.n_components)
        self.params = torch.cat((self.ys.view(-1), self.alphas.view(-1)))

    def forward(self, xyz):
        # TODO: Could be optimized
        cs = []
        for y, alpha in zip(self.ys, self.alphas):
            ci = 0.5*sphere_dist_th(y, xyz)**2 + alpha
            cs.append(ci)
        cs = torch.stack(cs)
        if self.gamma == None or self.gamma == 0.:
            z = cs.min(dim=0).values
        else:
            z = -self.gamma*(-cs/self.gamma).logsumexp(dim=0)
        return z


seeds = [8,9,2,31,4,20,16,7]
fs = [c_convex(seed=i) for i in seeds]
n_params = len(fs[0].params)

class AmortizedModel(nn.Module):
    def __init__(self, n_params):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(n_params, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, 3)
        )

    def forward(self, p):
        squeeze = p.ndim == 1
        if squeeze:
            p = p.unsqueeze(0)
        assert p.ndim == 2
        z = self.base(p)
        z = z / z.norm(dim=-1, keepdim=True)
        if squeeze:
            z = z.squeeze(0)
        return z

n_hidden = 128
torch.manual_seed(0)
model = AmortizedModel(n_params=n_params)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)

xs = []
for i in range(100):
    losses = []
    xis = []
    for f in fs:
        pred_opt = model(f.params)
        xis.append(pred_opt)
        losses.append(f(pred_opt))
    with torch.no_grad():
        xis = torch.stack(xis)
        xs.append(xis)
    loss = sum(losses)

    opt.zero_grad()
    loss.backward()
    opt.step()

xs = torch.stack(xs, dim=1)

pad = .1
n_sample = 100
ra = np.linspace(-np.pi+pad, np.pi-pad, n_sample)
dec= np.linspace(-np.pi/2+pad, np.pi/2-pad, n_sample)
ra_grid, dec_grid = np.meshgrid(ra,dec)
ra_grid_flat = ra_grid.ravel()
dec_grid_flat = dec_grid.ravel()
x_grid, y_grid, z_grid = celestial_to_euclidean(ra_grid_flat, dec_grid_flat)

p_grid = np.stack((x_grid, y_grid, z_grid), axis=-1)
p_grid_th = torch.from_numpy(p_grid).float()


for i, (f, xs_i) in enumerate(zip(fs, xs)):
    nrow, ncol = 1, 1
    fig, ax = plt.subplots(
        nrow, ncol, figsize=(3*ncol, 2*nrow),
        subplot_kw={'projection': 'mollweide'},
        gridspec_kw = {'wspace':0, 'hspace':0}
    )

    with torch.no_grad():
        f_grid = f(p_grid_th).numpy()
    best_i = f_grid.argmin()
    ra_opt, dec_opt= ra_grid_flat[best_i], dec_grid_flat[best_i]

    f_grid = f_grid.reshape(ra_grid.shape)
    n_levels = 10
    ax.contourf(ra_grid, dec_grid, f_grid, n_levels, cmap='Purples')

    x,y,z = xs_i.split(1,dim=-1)
    ra, dec = euclidean_to_celestial_th(x,y,z)
    ax.plot(ra, dec, color='#5499FF', lw=3, ls=':')

    ax.scatter(ra_opt, dec_opt, marker='*', color='#AA0000',
               s=100, zorder=10)

    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    fname = f'paper/fig/sphere/{i}.png'
    plt.savefig(fname, transparent=True)
    os.system(f'convert -trim {fname} {fname}')
