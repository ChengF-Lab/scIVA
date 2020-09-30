#!/usr/bin/env python
"""
#
#

# File Name: scIVA.py
# Description: Dimensionality reduction and visualization of single-cell RNA-seq data with an improved deep variational autoencoder.
    Input: 
        single-cell RNA-seq data
    Output:
        1. latent feature
        2. cluster assignment

"""


import time
import torch

import numpy as np
import pandas as pd
import os
import argparse

from sciva import scIVA
from sciva.dataset import SingleCellDataset
from sciva.utils import  estimate_k
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader
from sciva.helpers  import clustering, measure, print_2D

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dimensionality reduction and visualization of single-cell RNA-seq data with an improved deep variational autoencoder')
    parser.add_argument('--dataset', '-d', type=str, help='input dataset path', default="../scIVA-master/data/yan.txt")
    parser.add_argument('--n_centroids', '-k', type=int, help='cluster number')
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
    parser.add_argument('--pretrain', type=str, default=None, help='Load the trained model')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
    parser.add_argument('--encode_dim', type=int, nargs='*', default=[512,128,64], help='encoder structure')
    parser.add_argument('--decode_dim', type=int, nargs='*', default=[64,128,512], help='encoder structure')
    parser.add_argument('--latent', '-l',type=int, default=2, help='latent layer dim')
    parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
    parser.add_argument('--max_iter', '-i', type=int, default=20000, help='Max iteration')
    parser.add_argument('--weight_decay', type=float, default=10e-4)
    parser.add_argument('--reference', '-r',  type=str, help='Reference celltypes',default="../scIVA-master/data/yan_label.txt")
    parser.add_argument('--transpose', '-t', action='store_true', help='Transpose the input matrix')
    DATASET = '../scIVA-master/data/yan'  # sys.argv[1]
    filename = DATASET + '.txt'
    data = open(filename)
    head = data.readline().rstrip().split()
    #print(head)
    label_file = open(DATASET + '_label.txt')
    label_dict = {}
    for line in label_file:
        temp = line.rstrip().split()
        label_dict[temp[0]] = temp[1]
    label_file.close()

    label = []
    for c in head:
        if c in label_dict.keys():
            label.append(label_dict[c])
        else:
            print(c)

    label_set = []
    for c in label:
        if c not in label_set:
            label_set.append(c)
    name_map = {value: idx for idx, value in enumerate(label_set)}
    id_map = {idx: value for idx, value in enumerate(label_set)}
    label = np.asarray([name_map[name] for name in label])
    print(label)
    args = parser.parse_args()
    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(args.gpu)
    else:
        device='cpu'
    batch_size = args.batch_size

    normalizer = MaxAbsScaler()
    dataset = SingleCellDataset(args.dataset,
                                transpose=args.transpose, transforms=[normalizer.fit_transform])

    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    cell_num = dataset.shape[0] 
    input_dim = dataset.shape[1] 	
    
    if args.n_centroids is None:
        k = min(estimate_k(dataset.data.T), 15)
        print('Estimate k {}'.format(k))
    else:
        k = args.n_centroids
    lr = args.lr
    name = args.dataset.strip('/').split('/')[-1]

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    print("\n**********************************************************************")
    print("Dimensionality reduction and visualization of single-cell RNA-seq data with an improved deep variational autoencoder")
    print("**********************************************************************\n")

    dims = [input_dim, args.latent, args.encode_dim, args.decode_dim]
    model = scIVA(dims, n_centroids=k)
#     print(model)

    if not args.pretrain:
        print('\n## Training Model ##')
        model.init_gmm_params(testloader)
        model.fit(trainloader,
                  lr=lr, 
                  weight_decay=args.weight_decay,
                  verbose=args.verbose,
                  device = device,
                  max_iter=args.max_iter,
                  name=name,
                  outdir=outdir
                   )
#         torch.save(model.to('cpu').state_dict(), os.path.join(outdir, 'model.pt')) # save model
    else:
        print('\n## Loading Model: {}\n'.format(args.pretrain))
        model.load_model(args.pretrain)
        model.to(device)

    ### output ###
    print('outdir: {}'.format(outdir))
    # 1. latent feature
    feature = model.encodeBatch(testloader, device=device, out='z')
    pd.DataFrame(feature).to_csv(os.path.join(outdir, 'feature.txt'), sep='\t', header=False)

    fig = print_2D(points=feature, label=label, id_map=id_map)
    fig.savefig('embryo.eps')
    k = len(np.unique(label))
    cl, _ = clustering(feature, k=k)
    dm = measure(cl, label)
