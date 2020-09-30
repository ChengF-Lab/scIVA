#!/usr/bin/env python
"""
#
#

# File Name: layer.py
# Description:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

import math
import numpy as np


def build_mlp(layers, activation=nn.ReLU(), bn=False, dropout=0):
    """
    Build multilayer linear perceptron
    """
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class ZILayer(nn.Module):
    def __init__(self, init_t=1, min_t=0.5, anneal_rate=0.00001, annealing=False):
        super(ZILayer, self).__init__()
        self.init_t = init_t
        self.min_t = min_t
        self.anneal_rate = anneal_rate
        self.annealing = annealing
        self.iteration = nn.Parameter(torch.tensor(0, dtype=torch.int), requires_grad=False)
        self.temperature = nn.Parameter(torch.tensor(init_t, dtype=torch.float), requires_grad=False)

    def forward(self, probs):
        p = torch.exp(-(probs**2))
        q = 1-p
        logits = torch.log(torch.stack([p, q], dim=-1)+1e-20)
        g = self.sampling_gumbel(logits.shape).type_as(logits)
        samples = torch.softmax((logits+g)/self.temperature, dim=-1)[..., 1]
        output = probs * samples
        if self.training and self.annealing:
            self.adjust_temperature()
        # print(output.mean().item(), output.std().item(), probs.mean().item(), probs.std().item())
        return output

    def sampling_gumbel(self, shape, eps=1e-8):
        u = torch.rand(*shape)
        return -torch.log(-torch.log(u + eps) + eps)

    def adjust_temperature(self):
        self.iteration.data += 1
        if self.iteration % 100== 0:
            # print(self.temperature.item())
            t = torch.clamp(self.init_t * torch.exp(-self.anneal_rate * self.iteration), min=self.min_t)
            self.temperature.data = t



class Encoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(Encoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        # self.hidden = build_mlp([x_dim, *h_dim], bn=bn, dropout=dropout)
        # dropout (bn)的设定
        self.hidden = build_mlp([x_dim]+h_dim, bn=bn, dropout=dropout)
        self.sample = GaussianSample(([x_dim]+h_dim)[-1], z_dim)
        # self.sample = GaussianSample([x_dim, *h_dim][-1], z_dim)

    def forward(self, x):
        x = self.hidden(x);
        return self.sample(x)


class Decoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, output_activation=nn.Sigmoid()):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims
        # dropout (bn)的设定
        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
#         self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)
        self.reconstruction = nn.Linear([z_dim, *h_dim][-1], x_dim)
#         self.reconstruction = nn.Linear(([z_dim]+h_dim)[-1], x_dim)

        self.output_activation = output_activation
        self.zilayer = ZILayer(annealing=True)

    def forward(self, x):
        x = self.hidden(x)
        if self.output_activation is not None:
            res =  self.output_activation(self.reconstruction(x))
        else:
            res =  self.reconstruction(x)
        res = self.zilayer(res)
        return res

class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [Sønderby 2016].
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t

    def next(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t


###################
###################
class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device)
        std = logvar.mul(0.5).exp_()
#         std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return self.reparametrize(mu, log_var), mu, log_var

