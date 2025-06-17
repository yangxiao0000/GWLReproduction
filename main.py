"""
Matching communication network with email network in the MC3 dataset
"""
from utils import *
# import dev.util as util
import matplotlib.pyplot as plt
from model.GromovWassersteinLearning import GromovWassersteinLearning
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora', help='douban/dblp/cora/citeseer/facebook/ppi')
# parser.add_argument('--dataset', type=str, default='cora', help='douban/dblp/cora/citeseer/facebook/ppi')
parser.add_argument('--feat_noise', type=float, default=0.)
parser.add_argument('--noise_type', type=int, default=0, help='1: permutation, 2: truncation, 3: compression')
parser.add_argument('--edge_noise', type=float, default=0.)
parser.add_argument('--output', type=str, default='result.txt')
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--truncate', type=bool, default=False)


args = parser.parse_args()


# args.dataset='facebook'
# truncate= True
# edge_noise=0.1

cost_type = ['cosine']
method = ['proximal']
save_path = '/root/sharedata' 

adj1=torch.load(save_path + '/'+args.dataset+'Aadj.pt') 
adj2=torch.load(save_path + '/'+args.dataset+'Badj.pt')
f1=torch.load(save_path + '/'+args.dataset+'Afeat.pt')
f2=torch.load(save_path + '/'+args.dataset+'Bfeat.pt')
ground_truth=torch.load(save_path + '/'+args.dataset+'ground_truth.pt')
a1= adjacency_to_edge_matrix(adj1)
a2= adjacency_to_edge_matrix(adj2)
# print(a1.shape)
# exit(0)
#edge_noise
if args.edge_noise > 0:
    adj1 = torch.tensor(add_noise_edge(a1, adj1, args.edge_noise)) 
    adj2 = torch.tensor(add_noise_edge(a2, adj2, args.edge_noise))
    a1 = adjacency_to_edge_matrix(adj1)
    a2 = adjacency_to_edge_matrix(adj2)
#truncate
if args.truncate==True:
    f1=f1[:,:100]
    f2=f2[:,:100]
#feat_noise
num1=adj1.shape[0]
num2=adj2.shape[0]
index1= create_number_dict(num1)
index2= create_number_dict(num2)
inter1=convert_to_list(a1)
inter2=convert_to_list(a2)
dataset={}
dataset['src_index'] = index1
dataset['tar_index'] = index2
dataset['src_interactions'] = inter1
dataset['tar_interactions'] = inter2
dataset['ground_truth'] = ground_truth
dataset['mutual_interactions'] = ground_truth.tolist()


dataset['cost_s']=1/(adj1+1)
dataset['cost_t']=1/(adj2+1)
# print(dataset['cost_s'].shape, dataset['cost_t'].shape)
# exit(0)


opt_dict = {
            # 'epochs': 30,
            # 'epochs': 30,
            'epochs': args.epoch,
            # 'batch_size': 20000,
            'batch_size': 57000,
            # 'batch_size': 32316,
            'use_cuda': True,
            'strategy': 'hard',
            'beta': args.beta,
            # 'outer_iteration': 10000,
            'outer_iteration': 3000,
            # 'outer_iteration': 200,
            'inner_iteration': 1,
            # 'inner_iteration': 100,
            'sgd_iteration': 500,
            'prior': False,
            # 'prefix': result_folder,
            'display': False,
            'dataset': args.dataset,
            'feat_noise': args.feat_noise,
            'noise_type': args.noise_type,
            'edge_noise': args.edge_noise,}



hyperpara_dict = {'src_number': len(dataset['src_index']),
                    'tar_number': len(dataset['tar_index']),
                    'dimension': 50,
                    'loss_type': 'L2',
                    'cost_type': 'cosine',
                    'ot_method': 'proximal'}

gwd_model = GromovWassersteinLearning(hyperpara_dict)


optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

gwd_model.my_train_without_prior(dataset, optimizer, opt_dict, scheduler=None)




