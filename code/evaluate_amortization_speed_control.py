#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch

import argparse
import os
import sys
import pickle as pkl
import shutil
from omegaconf import OmegaConf
from collections import namedtuple
import dmc2gym

import matplotlib.pyplot as plt
plt.style.use('bmh')
from matplotlib import cm

from multiprocessing import Process

from svg.video import VideoRecorder
from svg import utils, dx

from evaluate_amortization_speed_function import evaluate_amortization_speed

def main():
    import sys
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                         color_scheme='Linux',
                                         call_pdb=1)

    exp = torch.load('svg_submodule/trained-humanoid/latest.pt')

    # Clean up logging after resuming the experiment code
    del exp.logger
    os.remove('eval.csv')
    os.remove('train.csv')

    observations = collect_eval_episode(exp)

    # First try to predict the maximum value action

    def amortization_model(observations):
        actions, _, _ = exp.agent.actor(observations, compute_pi=False, compute_log_pi=False)
        return actions

    def amortization_objective(actions, observations, normalize=True):
        q1, q2 = exp.agent.critic(observations, actions)
        values = torch.min(q1, q2).squeeze()
        if normalize:
            values = normalize_values(values)

        return values

    with torch.no_grad():
        expert_actions = amortization_model(observations)
        zero_actions = torch.zeros_like(expert_actions)
        expert_values = amortization_objective(expert_actions, observations, normalize=False)
        zero_values = amortization_objective(zero_actions, observations, normalize=False)

    def normalize_values(values):
        """normalize so that the expert value is 0 and the zero action is -1."""
        norm_values = (values - expert_values) / (expert_values - zero_values)

        # assume we can't do better than the expert.
        # otherwise the optimization overfits to the inaccurate model
        # and value approximation.
        norm_values[norm_values > 0.] = 0.
        return norm_values


    evaluate_amortization_speed(
        amortization_model=amortization_model,
        amortization_objective=amortization_objective,
        contexts=observations,
        tag='control-model-free',
        fig_ylabel='Value',
        adam_lr=5e-3,
        num_iterations=500,
        maximize=True,
    )

    # Next try to predict the solution to the short-horizon model-based
    # control problem.
    def amortization_model(observations):
        num_batch = observations.shape[0]
        action_seq, _, _ = exp.agent.dx.unroll_policy(
            observations, exp.agent.actor, sample=False, last_u=True)
        action_seq_flat = action_seq.transpose(0,1).reshape(num_batch, -1)
        return action_seq_flat

    def amortization_objective(action_seq_flat, observations, normalize=True):
        num_batch = action_seq_flat.shape[0]
        action_seq = action_seq_flat.reshape(num_batch, -1, exp.agent.action_dim).transpose(0, 1)
        predicted_states = exp.agent.dx.unroll(observations, action_seq[:-1])

        all_obs = torch.cat((observations.unsqueeze(0), predicted_states), dim=0)
        xu = torch.cat((all_obs, action_seq), dim=2)
        dones = exp.agent.done(xu).sigmoid().squeeze(dim=2)
        not_dones = 1. - dones
        not_dones = utils.accum_prod(not_dones)
        last_not_dones = not_dones[-1]

        rewards = not_dones * exp.agent.rew(xu).squeeze(2)
        q1, q2 = exp.agent.critic(all_obs[-1], action_seq[-1])
        q = torch.min(q1, q2).reshape(num_batch)
        rewards[-1] = last_not_dones * q

        rewards *= exp.agent.discount_horizon.unsqueeze(1)

        values = rewards.sum(dim=0)
        if normalize:
            values = normalize_values(values)
        return values

    with torch.no_grad():
        # used in the normalization
        expert_action_seq = amortization_model(observations)
        zero_action_seq = torch.zeros_like(expert_action_seq)
        expert_values = amortization_objective(expert_action_seq, observations, normalize=False)
        zero_values = amortization_objective(zero_action_seq, observations, normalize=False)

    evaluate_amortization_speed(
        amortization_model=amortization_model,
        amortization_objective=amortization_objective,
        contexts=observations,
        tag='control-model-based',
        fig_ylabel='Value',
        adam_lr=5e-3,
        num_iterations=500,
        maximize=True,
    )



def collect_eval_episode(exp):
    device = 'cuda'
    exp.env.set_seed(0)
    obs = exp.env.reset()
    done = False
    total_reward = 0.
    step = 0
    observations = []
    while not done:
        if exp.cfg.normalize_obs:
            mu, sigma = exp.replay_buffer.get_obs_stats()
            obs = (obs - mu) / sigma
        obs = torch.FloatTensor(obs).to(device)
        observations.append(obs)
        action, _, _ = exp.agent.actor(obs, compute_pi=False, compute_log_pi=False)
        action = action.clamp(min=exp.env.action_space.low.min(),
                              max=exp.env.action_space.high.max())

        obs, reward, done, _ = exp.env.step(utils.to_np(action.squeeze(0)))
        total_reward += reward
        step += 1
    print(f'+ eval episode reward: {total_reward}')
    observations = torch.stack(observations, dim=0)
    return observations


if __name__ == '__main__':
    main()
