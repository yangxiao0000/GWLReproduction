"""
Matching communication network with email network in the MC3 dataset
"""
from utils import *
# import dev.util as util
import matplotlib.pyplot as plt
# from model.GromovWassersteinLearning import GromovWassersteinLearning
import numpy as np
# import pickle
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

datasetname='facebook'
truncate= True
edge_noise=0.0

result_folder = 'match_mimic3_2'
cost_type = ['cosine']
method = ['proximal']
save_path = '/root/sharedata' 

adj1=torch.load(save_path + '/'+datasetname+'Aadj.pt') 
adj2=torch.load(save_path + '/'+datasetname+'Badj.pt')
f1=torch.load(save_path + '/'+datasetname+'Afeat.pt')
f2=torch.load(save_path + '/'+datasetname+'Bfeat.pt')
ground_truth=torch.load(save_path + '/'+datasetname+'ground_truth.pt')
a1= adjacency_to_edge_matrix(adj1)
a2= adjacency_to_edge_matrix(adj2)
# print(a1.shape)
# exit(0)
#edge_noise
if edge_noise > 0:
    adj1 = torch.tensor(add_noise_edge(a1, adj1, edge_noise)) 
    adj2 = torch.tensor(add_noise_edge(a2, adj2, edge_noise))
    a1 = adjacency_to_edge_matrix(adj1)
    a2 = adjacency_to_edge_matrix(adj2)
#truncate
if truncate==True:
    f1=f1[:,:100]
    f2=f2[:,:100]
#feat_noise

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


opt_dict = {
            # 'epochs': 30,
            'epochs': 1,
            # 'epochs': 5,
            # 'batch_size': 20000,
            'batch_size': 57000,
            # 'batch_size': 32316,
            'use_cuda': True,
            'strategy': 'soft',
            'beta': 1e-2,
            # 'outer_iteration': 10000,
            'outer_iteration': 10000,
            # 'outer_iteration': 200,
            'inner_iteration': 1,
            # 'inner_iteration': 100,
            'sgd_iteration': 500,
            'prior': False,
            'prefix': result_folder,
            'display': False}

for m in method:
    for c in cost_type:
        hyperpara_dict = {'src_number': len(dataset['src_index']),
                          'tar_number': len(dataset['tar_index']),
                          'dimension': 50,
                          'loss_type': 'L2',
                          'cost_type': c,
                          'ot_method': m}

        gwd_model = GromovWassersteinLearning(hyperpara_dict)

        # initialize optimizer
        optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=1e-3)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        # Gromov-Wasserstein learning
        # gwd_model.train_with_prior(data_mc3, optimizer, opt_dict, scheduler=None)
        gwd_model.train_without_prior(data_mc3, optimizer, opt_dict, scheduler=None)
        
        # save model
        gwd_model.save_model('{}/model_{}_{}.pt'.format(result_folder, m, c))
        gwd_model.save_recommend('{}/result_{}_{}.pkl'.format(result_folder, m, c))



