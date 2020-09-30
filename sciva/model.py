#!/usr/bin/env python
"""
#
#

# File Name: model.py
# Description:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau

import time
import math
import numpy as np
from tqdm import trange
from itertools import repeat
from sklearn.mixture import GaussianMixture

from .layer import Encoder, Decoder, build_mlp, DeterministicWarmup
from .loss import elbo, elbo_scIVA


class VAE(nn.Module):
    def __init__(self, dims, bn=True, dropout=0.15, binary=True):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VAE, self).__init__()
        [x_dim, z_dim, encode_dim, decode_dim] = dims
        self.binary = binary
        if binary:
            decode_activation = nn.Sigmoid()
        else:
            decode_activation = None

        self.encoder = Encoder([x_dim, encode_dim, z_dim], bn=bn, dropout=dropout)
        self.decoder = Decoder([z_dim, decode_dim, x_dim], bn=bn, dropout=dropout, output_activation=decode_activation)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)

        return recon_x

    def loss_function(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        likelihood, kld = elbo(recon_x, x, (mu, logvar), binary=self.binary)

        return (-likelihood, kld)


    def predict(self, dataloader, device='cpu', method='kmeans'):
        """
        Predict assignments applying k-means on latent feature

        Input: 
            x, data matrix
        Return:
            predicted cluster assignments
        """
        self.eval()

        if method == 'kmeans':
            from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
            feature = self.encodeBatch(dataloader, device)
            kmeans = KMeans(n_clusters=self.n_centroids, n_init=20, random_state=0)
            pred = kmeans.fit_predict(feature)
        elif method == 'gmm':
            logits = self.encodeBatch(dataloader, device, out='logit')
            pred = np.argmax(logits, axis=1)

        return pred

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def fit(self, dataloader,
            lr=0.00001,
            weight_decay=5e-4,
            device='cpu',
            beta = 1,
            n = 200,
            max_iter=30000,
            verbose=True,
            name='',
            patience=100,
            outdir='./'
       ):

        self.to(device)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        Beta = DeterministicWarmup(n=n, t_max=beta)
        
        iteration = 0
        early_stopping = EarlyStopping(patience=patience, outdir=outdir)
        with trange(max_iter, disable=verbose) as pbar:
            while True: 
                epoch_loss = 0
                for i, x in enumerate(dataloader):
                    epoch_lr = adjust_learning_rate(lr, optimizer, iteration)
                    t0 = time.time()
                    x = x.float().to(device)
                    optimizer.zero_grad()
                    
                    recon_loss, kl_loss = self.loss_function(x)
                    loss = (recon_loss + next(Beta) * kl_loss)/len(x);
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    pbar.set_postfix_str('loss={:.3f} recon_loss={:.3f} kl_loss={:.3f}'.format(
                            loss, recon_loss/len(x), kl_loss/len(x)))
                    pbar.update(1)
                    
                    iteration+=1
                    if iteration >= max_iter:
                        break
                else:
                    early_stopping(epoch_loss, self)
                    if early_stopping.early_stop:
                        print('EarlyStopping: run {} iteration'.format(iteration))
                        break
                    continue
                break

    def encodeBatch(self, dataloader, device='cpu', out='z', transforms=None):
        self.eval()
        output = []
        for i, inputs in enumerate(dataloader):
            x = inputs
            x = x.view(x.size(0), -1).float().to(device)
            z, mu, logvar = self.encoder(x)

            if out == 'z':
                output.append(z.detach().cpu())
            elif out == 'x':
                recon_x = self.decoder(z)
                output.append(recon_x.detach().cpu().data)
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach())

        output = torch.cat(output).numpy()
        if out == 'x':
            for transform in transforms:
                output = transform(output)

        return output



class scIVA(VAE):
    def __init__(self, dims, n_centroids):
        super(scIVA, self).__init__(dims)
        self.n_centroids = n_centroids
        z_dim = dims[1]
        print(dims)
        # init c_params
        self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
        self.mu_c = nn.Parameter(torch.zeros(z_dim, n_centroids)) # mu
        self.var_c = nn.Parameter(torch.ones(z_dim, n_centroids)) # sigma^2

    def loss_function(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        gamma, mu_c, var_c, pi = self.get_gamma(z) #, self.n_centroids, c_params)
        likelihood, kld = elbo_scIVA(recon_x, x, gamma, (mu_c, var_c, pi), (mu, logvar), binary=self.binary)

        return -likelihood, kld

    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N,1) # NxK
        mu_c = self.mu_c.repeat(N,1,1) # NxDxK
        var_c = self.var_c.repeat(N,1,1) # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

    def init_gmm_params(self, dataloader, device='cpu'):
        """
        Init scIVA model with GMM model parameters
        """
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        z = self.encodeBatch(dataloader, device)

        gmm.fit(z)
        print(gmm.means_.T.shape)
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))


def adjust_learning_rate(init_lr, optimizer, iteration):
    lr = max(init_lr * (0.9 ** (iteration//10)), 0.0002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr	


import os
class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, outdir='./'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = os.path.join(outdir, 'model.pt')

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss
