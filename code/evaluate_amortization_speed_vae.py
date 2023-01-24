#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
import numpy as np

import argparse
import os
import sys

sys.path.append('vae_submodule')
from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed
from utils.visualize import Visualizer
from utils.viz_helpers import get_samples
from disvae.utils.modelIO import load_model, load_metadata
from disvae.models.losses import get_loss_f

from evaluate_amortization_speed_function import evaluate_amortization_speed

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)


def sample_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + std * eps

def unflatten_latent(z_flat):
    n = z_flat.shape[-1]
    return z_flat[...,:n//2], z_flat[...,n//2:]


def estimate_elbo(x, z_flat, decoder):
    latent_dist = unflatten_latent(z_flat)

    latent_sample = sample_gaussian(*latent_dist)
    latent_sample = latent_sample
    recon_batch = decoder(latent_sample)
    batch_size = x.shape[0]
    log_likelihood = -F.binary_cross_entropy(recon_batch, x, reduce=False).sum(dim=[1,2,3])

    mean, logvar = latent_dist
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp())
    kl_to_prior = latent_kl.sum(dim=[-1])

    assert log_likelihood.shape == kl_to_prior.shape
    loss = log_likelihood - kl_to_prior
    return loss


def main():
    model_dir = 'vae_submodule/results/VAE_mnist'
    meta_data = load_metadata(model_dir)
    model = load_model(model_dir).cuda()
    model.eval()  # don't sample from latent: use mean
    dataset = meta_data['dataset']
    loss_f = get_loss_f('VAE',
                        n_data=len(dataset),
                        device='cuda',
                        rec_dist='bernoulli',
                        reg_anneal=0)

    batch_size = 1024
    num_save = 15
    data_samples = get_samples(dataset, batch_size, idcs=[25518, 13361, 22622]).cuda()

    def amortization_model(data_samples):
        latent_dist = model.encoder(data_samples)
        latent_dist_flat = torch.cat(latent_dist, dim=-1)
        return latent_dist_flat

    def amortization_objective(latent_dist_flat, data_samples):
        elbo = estimate_elbo(data_samples, latent_dist_flat, model.decoder)
        return elbo

    iterate_history, predicted_samples = evaluate_amortization_speed(
        amortization_model=amortization_model,
        amortization_objective=amortization_objective,
        contexts=data_samples,
        tag='vae',
        fig_ylabel='ELBO',
        adam_lr=5e-3,
        num_iterations=2000,
        maximize=True,
        save_iterates=[0, 250, 500, 1000, 2000],
        num_save=num_save,
    )

    iterate_history.append((-1, predicted_samples[:num_save]))

    reconstructions = []
    for i, latent_dist_flat in iterate_history:
        latent_dist = unflatten_latent(latent_dist_flat)
        latent_mean = latent_dist[0]
        reconstructions.append(1.-model.decoder(latent_mean))

    reconstructions.append(1.-data_samples[:num_save])
    reconstructions = torch.cat(reconstructions, dim=0)
    reconstructions = F.interpolate(reconstructions,
                                    recompute_scale_factor=True, scale_factor=1.5, mode='bilinear')

    fname = f'vae-samples.png'
    save_image(reconstructions, fname, nrow=num_save)


if __name__ == '__main__':
    main()
