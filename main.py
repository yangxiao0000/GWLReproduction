"""
Matching communication network with email network in the MC3 dataset
"""

import dev.util as util
import matplotlib.pyplot as plt
from model.GromovWassersteinLearning import GromovWassersteinLearning
import numpy as np
import pickle
import torch.optim as optim
from torch.optim import lr_scheduler

dataset='facebook'
truncate= True
edge_noise=0.0
##############################################################################
def create_number_dict(n):
    """
    创建一个字典，其中键为字符串形式的数字，值为对应的整数
    
    参数:
    n (int): 正整数，表示字典的大小
    
    返回:
    dict: 格式为 {'1': 1, '2': 2, ..., 'n': n} 的字典
    """
    return {str(float(i)): i for i in range(1, n+1)}

import torch

def convert_to_list(tensor):
    if tensor.ndim != 2 or tensor.shape[0] != 2:
        raise ValueError("输入必须是一个 2×N 的张量")
    return tensor.transpose(0, 1).tolist()
###############################################################################

############################################################################################
def adjacency_to_edge_matrix(adj):
    """
    将邻接矩阵转换为边矩阵（不包含自环）
    
    参数:
        adj (torch.Tensor): M×M的邻接矩阵（支持有向/无向，支持权重/非权重）
    
    返回:
        torch.Tensor: 2×M'的边矩阵，其中M'是非自环边的数量
    """
    # 获取非零元素的索引（包括自环）
    all_edges = torch.nonzero(adj).t()
    
    # 过滤掉自环边 (i == j)
    non_self_loops = all_edges[:, all_edges[0] != all_edges[1]]
    
    return non_self_loops

import random
def add_noise_edge(Aedge, Aadj, edge_noise):
    Aadjn = np.zeros_like(Aadj)
    for u, v in zip(Aedge[0], Aedge[1]):
        if u==v:
            Aadjn[u][v] = 1
        if u<v:
            if random.random() > edge_noise:
                Aadjn[u][v], Aadjn[v][u] = 1, 1
            else:
                while 1:
                    u, v = random.randint(0, Aadj.shape[0]-1), random.randint(0, Aadj.shape[0]-1)
                    if u != v and Aadj[u][v] == 0:
                        Aadjn[u][v], Aadjn[v][u] = 1, 1
                        break
    return Aadjn


############################################################################################



data_name = 'mimic3_2'
result_folder = 'match_mimic3_2'
cost_type = ['cosine']
method = ['proximal']

filename = '{}/{}_database.pickle'.format(util.DATA_TRAIN_DIR, data_name)
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    data_mc3 = pickle.load(f)

print(len(data_mc3['src_index']))
print(len(data_mc3['tar_index']))
print(len(data_mc3['src_interactions']))
print(len(data_mc3['tar_interactions']))



############################################################################################################

# print(a1.shape, f1.shape, a2.shape, f2.shape, ground_truth.shape)
# save_path = '/root/sharedata' 
# Aadj=torch.load(save_path + '/Aadj.pt') 
# Badj=torch.load(save_path + '/Badj.pt')
# Afeat=torch.load(save_path + '/Afeat.pt')
# Bfeat=torch.load(save_path + '/Bfeat.pt')
# gt=torch.load(save_path + '/ground_truth.pt')
# Aedge= adjacency_to_edge_matrix(Aadj)
# Bedge= adjacency_to_edge_matrix(Badj)
# print(Aadj.shape, Badj.shape, Afeat.shape, Bfeat.shape, gt.shape)
save_path = '/root/sharedata' 
adj1=torch.load(save_path + '/'+dataset+'Aadj.pt') 
adj2=torch.load(save_path + '/'+dataset+'Badj.pt')
f1=torch.load(save_path + '/'+dataset+'Afeat.pt')
f2=torch.load(save_path + '/'+dataset+'Bfeat.pt')
ground_truth=torch.load(save_path + '/'+dataset+'ground_truth.pt')
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

num1=adj1.shape[0]
num2=adj2.shape[0]
index1= create_number_dict(num1)
index2= create_number_dict(num2)
data_mc3['src_index'] = index1
data_mc3['tar_index'] = index2
# print(data_mc3['src_index'])
# print(a1)
inter1=convert_to_list(a1)
inter2=convert_to_list(a2)
data_mc3['src_interactions'] = inter1
data_mc3['tar_interactions'] = inter2
# print(data_mc3['src_interactions'])


# ground_truth=ground_truth.numpy()
# print(ground_truth.shape)
data_mc3['ground_truth'] = ground_truth
# print(data_mc3['mutual_interactions'])
data_mc3['mutual_interactions'] = ground_truth.tolist()
# data_mc3['mutual_interactions'] = None
# print(data_mc3['mutual_interactions'])

# data_mc3['src_number']=num1
# data_mc3['tar_number']=num2
print(a1.shape)
print(len(data_mc3['src_index']))

print(len(data_mc3['src_interactions']))
# exit(0)
# print(data_mc3.keys())
# print(data_mc3['src_title'])
# exit(0)
# print(data_mc3['ground_truth'].shape)





# print(a1)
# print(Aedge.shape)
# print(gt)
# print(ground_truth)


# print(a1.shape, f1.shape, a2.shape, f2.shape,ground_truth.shape)
############################################################################################################
# print(index1)/


# exit(0)




# connects = np.zeros((len(data_mc3['src_index']), len(data_mc3['tar_index'])))
# for item in data_mc3['src_interactions']:
#     connects[item[0], item[1]] += 1
# plt.imshow(connects)
# plt.savefig('{}/src.png'.format(result_folder))
# plt.close('all')
#
# connects = np.zeros((len(data_mc3['src_index']), len(data_mc3['tar_index'])))
# for item in data_mc3['tar_interactions']:
#     connects[item[0], item[1]] += 1
# plt.imshow(connects)
# plt.savefig('{}/tar.png'.format(result_folder))
# plt.close('all')

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
        hyperpara_dict = {'src_number': len(data_mc3['src_index']),
                          'tar_number': len(data_mc3['tar_index']),
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



