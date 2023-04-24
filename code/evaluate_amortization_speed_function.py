# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('bmh')
params = {
    "text.usetex" : True,
    "font.family" : "serif",
    "font.serif" : ["Computer Modern Serif"]
}
plt.rcParams.update(params)

import os
import time

def evaluate_amortization_speed(
        amortization_model,
        amortization_objective,
        contexts,
        tag,
        fig_ylabel,
        adam_lr=5e-3,
        num_iterations=2000,
        maximize=False,
        iter_history_callback=None,
        save_iterates=[],
        num_save=8,
):
    times = []
    n_trials = 10
    for i in range(n_trials+1):
        start_time = time.time()
        predicted_solutions = amortization_model(contexts)
        if i > 0:
            times.append(time.time()-start_time)

    amortized_objectives = amortization_objective(
        predicted_solutions, contexts
    ).cpu().detach()
    print(f'solution size: {predicted_solutions.shape[1]}')
    print('--- amortization model')
    print(f'average objective value: {amortized_objectives.mean():.2f}')
    print(f'average runtime: {np.mean(times)*1000:.2f}ms')

    iterates = torch.nn.Parameter(torch.zeros_like(predicted_solutions))

    opt = torch.optim.Adam([iterates], lr=adam_lr)

    objective_history = []
    times = []
    iterations = []
    iterate_history = []

    start_time = time.time()

    for i in range(num_iterations+1):
        objectives = amortization_objective(iterates, contexts)
        mean_objective = objectives.mean()
        if maximize:
            mean_objective *= -1.
        opt.zero_grad()
        mean_objective.backward()
        opt.step()

        if i % 50 == 0:
            iterations.append(i)
            times.append(time.time()-start_time)
            objective_history.append((objectives.mean().item(), objectives.std().item()))
            print(i, objectives.mean().item())

        if i in save_iterates:
            iterate_history.append((i, iterates[:num_save].detach().clone()))


    times = np.array(times)

    figsize = (4,2)
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    objective_means, objective_stds = map(np.array, zip(*objective_history))

    l, = ax.plot(iterations, objective_means)
    ax.axhline(amortized_objectives.mean().cpu().detach(), color='k', linestyle='--')
    ax.axhspan(amortized_objectives.mean()-amortized_objectives.std(),
               amortized_objectives.mean()+amortized_objectives.std(), color='k', alpha=0.15)
    ax.fill_between(
        iterations, objective_means-objective_stds, objective_means+objective_stds,
        color=l.get_color(), alpha=0.5)
    ax.set_xlabel('Adam Iterations')
    ax.set_ylabel(fig_ylabel)
    ax.set_xlim(0, max(iterations))
    # ax.set_ylim(0, 1000)
    fig.tight_layout()
    fname = f'{tag}-iter.pdf'
    print(f'saving to {fname}')
    fig.savefig(fname, transparent=True)
    os.system(f'pdfcrop {fname} {fname}')

    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.axhline(amortized_objectives.mean(), color='k', linestyle='--')
    ax.axhspan(amortized_objectives.mean()-amortized_objectives.std(),
               amortized_objectives.mean()+amortized_objectives.std(), color='k', alpha=0.15)
    l, = ax.plot(times, objective_means)
    ax.fill_between(
        times, objective_means-objective_stds, objective_means+objective_stds,
        color=l.get_color(), alpha=0.5)
    ax.set_xlim(0, max(times))
    # ax.set_ylim(0, 1000)
    ax.set_xlabel('Runtime (seconds)')
    ax.set_ylabel(fig_ylabel)
    fig.tight_layout()

    fname = f'{tag}-time.pdf'
    print(f'saving to {fname}')
    fig.savefig(fname, transparent=True)
    os.system(f'pdfcrop {fname} {fname}')

    return iterate_history, predicted_solutions

