"""
This script contains the functions related to Gromove-Wasserstein Learning
"""
import copy
from dev.util import logger
import matplotlib.pyplot as plt
import numpy as np
import pickle
from preprocess.DataIO import IndexSampler, cost_sampler1, cost_sampler2
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
#############################################################################
def my_check_align(pred, ground_truth, result_file=None):
    g_map = {}
    for i in range(ground_truth.size(1)):
    # for i in range(ground_truth.shape(1)):
        g_map[ground_truth[1, i].item()] = ground_truth[0, i].item()
    g_list = list(g_map.keys())
    ind = (-pred).argsort(axis=1)[:, :30]
    a1, a5, a10, a30 = 0, 0, 0, 0
    for i, node in enumerate(g_list):
        for j in range(30):
            if j >= pred.shape[1]:
                break
            if ind[node, j].item() == g_map[node]:
                if j < 1:
                    a1 += 1
                if j < 5:
                    a5 += 1
                if j < 10:
                    a10 += 1
                if j < 30:
                    a30 += 1
    a1 /= len(g_list)
    a5 /= len(g_list)
    a10 /= len(g_list)
    a30 /= len(g_list)
    # print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@30 %.2f%%' % (a1 * 100, a5 * 100, a10*100, a30*100))
    return a1,a5,a10,a30
#############################################################################

class GromovWassersteinEmbedding(nn.Module):
    """
    Learning embeddings from Cosine similarity
    """
    def __init__(self, num1: int, num2: int, dim: int, cost_type: str = 'cosine', loss_type: str = 'L2'):
        super(GromovWassersteinEmbedding, self).__init__()
        self.num1 = num1
        self.num2 = num2
        self.dim = dim
        self.cost_type = cost_type
        self.loss_type = loss_type
        emb1 = nn.Embedding(self.num1, self.dim)
        emb1.weight = nn.Parameter(
            torch.FloatTensor(self.num1, self.dim).uniform_(-1 / self.dim, 1 / self.dim))
        emb2 = nn.Embedding(self.num2, self.dim)
        emb2.weight = nn.Parameter(
            torch.FloatTensor(self.num2, self.dim).uniform_(-1 / self.dim, 1 / self.dim))
        self.emb_model = nn.ModuleList([emb1, emb2])

    def orthogonal(self, index, idx):
        embs = self.emb_model[idx](index)
        orth = torch.matmul(torch.t(embs), embs)
        orth -= torch.eye(embs.size(1)).cuda()
        return (orth**2).sum()

    def self_cost_mat(self, index, idx):
        embs = self.emb_model[idx](index)  # (batch_size, dim)
        if self.cost_type == 'cosine':
            # cosine similarity
            energy = torch.sqrt(torch.sum(embs ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            cost = 1-torch.exp(-5*(1-torch.matmul(embs, torch.t(embs)) / (torch.matmul(energy, torch.t(energy)) + 1e-5)))
        else:
            # Euclidean distance
            embs = torch.matmul(embs, torch.t(embs))  # (batch_size, batch_size)
            embs_diag = torch.diag(embs).view(-1, 1).repeat(1, embs.size(0))  # (batch_size, batch_size)
            cost = 1-torch.exp(-(embs_diag + torch.t(embs_diag) - 2 * embs)/embs.size(1))
        return cost

    def mutual_cost_mat(self, index1, index2):
        embs1 = self.emb_model[0](index1)  # (batch_size1, dim)
        embs2 = self.emb_model[1](index2)  # (batch_size2, dim)
        if self.cost_type == 'cosine':
            # cosine similarity
            energy1 = torch.sqrt(torch.sum(embs1 ** 2, dim=1, keepdim=True))  # (batch_size1, 1)
            energy2 = torch.sqrt(torch.sum(embs2 ** 2, dim=1, keepdim=True))  # (batch_size2, 1)
            cost = 1-torch.exp(-(1-torch.matmul(embs1, torch.t(embs2))/(torch.matmul(energy1, torch.t(energy2))+1e-5)))
        else:
            # Euclidean distance
            embs = torch.matmul(embs1, torch.t(embs2))  # (batch_size1, batch_size2)
            # (batch_size1, batch_size2)
            embs_diag1 = torch.diag(torch.matmul(embs1, torch.t(embs1))).view(-1, 1).repeat(1, embs2.size(0))
            # (batch_size2, batch_size1)
            embs_diag2 = torch.diag(torch.matmul(embs2, torch.t(embs2))).view(-1, 1).repeat(1, embs1.size(0))
            cost = 1-torch.exp(-(embs_diag1 + torch.t(embs_diag2) - 2 * embs)/embs1.size(1))
        return cost

    def tensor_times_mat(self, cost_s, cost_t, trans, mu_s, mu_t):
        if self.loss_type == 'L2':
            # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = torch.matmul(cost_s ** 2, mu_s).repeat(1, trans.size(1))
            f2_st = torch.matmul(torch.t(mu_t), torch.t(cost_t ** 2)).repeat(trans.size(0), 1)
            cost_st = f1_st + f2_st
            cost = cost_st - 2 * torch.matmul(torch.matmul(cost_s, trans), torch.t(cost_t))
        else:
            # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = torch.matmul(cost_s * torch.log(cost_s + 1e-5) - cost_s, mu_s).repeat(1, trans.size(1))
            f2_st = torch.matmul(torch.t(mu_t), torch.t(cost_t)).repeat(trans.size(0), 1)
            cost_st = f1_st + f2_st
            cost = cost_st - torch.matmul(torch.matmul(cost_s, trans), torch.t(torch.log(cost_t + 1e-5)))
        return cost

    def similarity(self, cost_pred, cost_truth, mask=None):
        if mask is None:
            if self.loss_type == 'L2':
                loss = ((cost_pred - cost_truth) ** 2) * torch.exp(-cost_truth)
            else:
                loss = cost_pred * torch.log(cost_pred / (cost_truth + 1e-5))
        else:
            if self.loss_type == 'L2':
                # print(mask.size())
                # print(cost_truth.size())
                # print(cost_pred.size())
                loss = mask.data * ((cost_pred - cost_truth) ** 2) * torch.exp(-cost_truth)
            else:
                loss = mask.data * (cost_pred * torch.log(cost_pred / (cost_truth + 1e-5)))
        loss = loss.sum()
        return loss

    def forward(self, index1, index2, trans, mu_s, mu_t, cost1, cost2, prior=None, mask1=None, mask2=None, mask12=None):
        cost_s = self.self_cost_mat(index1, 0)
        cost_t = self.self_cost_mat(index2, 1)
        cost_st = self.mutual_cost_mat(index1, index2)
        cost = self.tensor_times_mat(cost_s, cost_t, trans, mu_s, mu_t)
        d_gw = (cost * trans).sum()
        d_w = (cost_st * trans).sum()
        regularizer = self.similarity(cost_s, cost1, mask1) + self.similarity(cost_t, cost2, mask2)
        regularizer += self.orthogonal(index1, 0) + self.orthogonal(index2, 1)
        if prior is not None:
            regularizer += self.similarity(cost_st, prior, mask12)
        return d_gw, d_w, regularizer

    def plot_and_save(self, index1: torch.Tensor, index2: torch.Tensor, output_name: str = None):
        """
        Plot and save cost matrix

        Args:
            index1: a (batch_size, 1) Long/CudaLong Tensor indicating the indices of entities
            index2: a (batch_size, 1) Long/CudaLong Tensor indicating the indices of entities
            output_name: a string indicating the output image's name
        Returns:
            save cost matrix as a .png file
        """
        cost_s = self.self_cost_mat(index1, 0).data.cpu().numpy()
        cost_t = self.self_cost_mat(index2, 0).data.cpu().numpy()
        cost_st = self.mutual_cost_mat(index1, index2).data.cpu().numpy()

        pc_kwargs = {'rasterized': True, 'cmap': 'viridis'}
        fig, axs = plt.subplots(1, 3, figsize=(5, 5), constrained_layout=True)

        im = axs[0, 0].pcolormesh(cost_s, **pc_kwargs)
        fig.colorbar(im, ax=axs[0, 0])
        axs[0, 0].set_title('source cost')
        axs[0, 0].set_aspect('equal')

        im = axs[0, 1].pcolormesh(cost_t, **pc_kwargs)
        fig.colorbar(im, ax=axs[0, 1])
        axs[0, 1].set_title('target cost')
        axs[0, 1].set_aspect('equal')

        im = axs[0, 2].pcolormesh(cost_st, **pc_kwargs)
        fig.colorbar(im, ax=axs[0, 2])
        axs[0, 2].set_title('mutual cost')
        axs[0, 2].set_aspect('equal')

        if output_name is None:
            plt.savefig('result.png')
        else:
            plt.savefig(output_name)
        plt.close("all")


class GromovWassersteinLearning(object):
    """
    Learning Gromov-Wasserstein distance in a nonparametric way.
    """
    def __init__(self, hyperpara_dict):
        """
        Initialize configurations

        Args:
            hyperpara_dict: a dictionary containing the configurations of model
                dict = {'src_number': the number of entities in the source domain,
                        'tar_number': the number of entities in the target domain,
                        'dimension': the proposed dimension of entities' embeddings,
                        'loss_type': 'KL' or 'L2'
                        }
        """
        self.src_num = hyperpara_dict['src_number']
        self.tar_num = hyperpara_dict['tar_number']
        self.dim = hyperpara_dict['dimension']
        self.loss_type = hyperpara_dict['loss_type']
        self.cost_type = hyperpara_dict['cost_type']
        self.ot_method = hyperpara_dict['ot_method']
        self.gwl_model = GromovWassersteinEmbedding(self.src_num, self.tar_num, self.dim, self.loss_type)
        self.d_gw = []
        self.trans = np.zeros((self.src_num, self.tar_num))
        self.Prec = []
        self.Recall = []
        self.F1 = []
        self.NC1 = []
        self.NC2 = []
        self.EC1 = []
        self.EC2 = []

    def plot_result(self, index_s, index_t, epoch, prefix):
        # tsne
        embs_s = self.gwl_model.emb_model[0](index_s)
        embs_t = self.gwl_model.emb_model[1](index_t)
        embs = np.concatenate((embs_s.cpu().data.numpy(), embs_t.cpu().data.numpy()), axis=0)
        embs = TSNE(n_components=2).fit_transform(embs)
        plt.figure(figsize=(5, 5))
        plt.scatter(embs[:embs_s.size(0), 0], embs[:embs_s.size(0), 1],
                    marker='.', s=0.5, c='b', edgecolors='b', label='graph 1')
        plt.scatter(embs[-embs_t.size(0):, 0], embs[-embs_t.size(0):, 1],
                    marker='o', s=8, c='r', edgecolors='r', label='graph 2')
        leg = plt.legend(loc='upper left', ncol=1, shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.title('T-SNE of node embeddings')
        plt.savefig('{}/emb_epoch{}_{}_{}.pdf'.format(prefix, epoch, self.ot_method, self.cost_type))
        plt.close("all")

        trans_b = np.zeros(self.trans.shape)
        for i in range(trans_b.shape[0]):
            idx = np.argmax(self.trans[i, :])
            trans_b[i, idx] = 1
        plt.imshow(trans_b)
        plt.savefig('{}/trans_epoch{}_{}_{}.png'.format(prefix, epoch, self.ot_method, self.cost_type))
        plt.close('all')


    def regularized_gromov_wasserstein_discrepancy(self, cost_s, cost_t, cost_mutual, mu_s, mu_t, hyperpara_dict):
        """
        Learning optimal transport from source to target domain

        Args:
            cost_s: (Ns, Ns) matrix representing the relationships among source entities
            cost_t: (Nt, Nt) matrix representing the relationships among target entities
            cost_mutual: (Ns, Nt) matrix representing the prior of proposed optimal transport
            mu_s: (Ns, 1) vector representing marginal probability of source entities
            mu_t: (Nt, 1) vector representing marginal probability of target entities
            hyperpara_dict: a dictionary of hyperparameters
                dict = {epochs: the number of epochs,
                        batch_size: batch size,
                        use_cuda: use cuda or not,
                        strategy: hard or soft,
                        beta: the weight of proximal term
                        outer_iter: the outer iteration of ipot
                        inner_iter: the inner iteration of sinkhorn
                        prior: True or False
                        }

        Returns:

        """
        ns = mu_s.size(0)
        nt = mu_t.size(0)
        trans = torch.matmul(mu_s, torch.t(mu_t))
        a = mu_s.sum().repeat(ns, 1)
        a /= a.sum()
        b = 0
        beta = hyperpara_dict['beta']

        if self.loss_type == 'L2':
            # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = torch.matmul(cost_s ** 2, mu_s).repeat(1, nt)
            f2_st = torch.matmul(torch.t(mu_t), torch.t(cost_t ** 2)).repeat(ns, 1)
            cost_st = f1_st + f2_st
            for t in range(hyperpara_dict['outer_iteration']):
                cost = cost_st - 2 * torch.matmul(torch.matmul(cost_s, trans), torch.t(cost_t)) + 0.1*cost_mutual
                if self.ot_method == 'proximal':
                    kernel = torch.exp(-cost / beta) * trans
                else:
                    kernel = torch.exp(-cost / beta)
                for l in range(hyperpara_dict['inner_iteration']):
                    b = mu_t / torch.matmul(torch.t(kernel), a)
                    a = mu_s / torch.matmul(kernel, b)
                    # print((b**2).sum())
                    # print((a**2).sum())
                    # print((b**2).sum()*(a**2).sum())
                trans = torch.matmul(torch.matmul(torch.diag(a[:, 0]), kernel), torch.diag(b[:, 0]))
                if t % 100 == 0:
                    print('sinkhorn iter {}/{}'.format(t, hyperpara_dict['outer_iteration']))
            cost = cost_st - 2 * torch.matmul(torch.matmul(cost_s, trans), torch.t(cost_t))

        else:
            # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = torch.matmul(cost_s * torch.log(cost_s + 1e-5) - cost_s, mu_s).repeat(1, nt)
            f2_st = torch.matmul(torch.t(mu_t), torch.t(cost_t)).repeat(ns, 1)
            cost_st = f1_st + f2_st
            for t in range(hyperpara_dict['outer_iteration']):
                cost = cost_st - torch.matmul(torch.matmul(cost_s, trans), torch.t(torch.log(cost_t + 1e-5)))
                if self.ot_method == 'proximal':
                    kernel = torch.exp(-cost / beta) * trans
                else:
                    kernel = torch.exp(-cost / beta)
                for l in range(hyperpara_dict['inner_iteration']):
                    b = mu_t / torch.matmul(torch.t(kernel), a)
                    a = mu_s / torch.matmul(kernel, b)
                trans = torch.matmul(torch.matmul(torch.diag(a[:, 0]), kernel), torch.diag(b[:, 0]))
            cost = cost_st - torch.matmul(torch.matmul(cost_s, trans), torch.t(torch.log(cost_t + 1e-5)))

        d_gw = (cost * trans).sum()
        return trans, d_gw, cost


    def my_train_without_prior(self, database, optimizer, hyperpara_dict, scheduler=None):
        """
        Regularized Gromov-Wasserstein Embedding
        Args:
            database: proposed database
            optimizer: the pytorch optimizer
            hyperpara_dict: a dictionary of hyperparameters
                dict = {epochs: the number of epochs,
                        batch_size: batch size,
                        use_cuda: use cuda or not,
                        strategy: hard or soft,
                        beta: the weight of proximal term
                        outer_iter: the outer iteration of ipot
                        inner_iter: the inner iteration of sinkhorn
                        prior: True or False
                        }
            scheduler: scheduler of learning rate.
        Returns:
            d_gw, trans
        """
        device = torch.device('cuda:0' if hyperpara_dict['use_cuda'] else 'cpu')
        if hyperpara_dict['use_cuda']:
            torch.cuda.manual_seed(1)
        kwargs = {'num_workers': 1, 'pin_memory': True} if hyperpara_dict['use_cuda'] else {}

        self.gwl_model.to(device)
        self.gwl_model.train()
        num_src_node = len(database['src_interactions'])
        num_tar_node = len(database['tar_interactions'])
        # src_loader = DataLoader(IndexSampler(num_src_node),
        #                         batch_size=hyperpara_dict['batch_size'],
        #                         shuffle=True,
        #                         **kwargs)
        # tar_loader = DataLoader(IndexSampler(num_tar_node),
        #                         batch_size=hyperpara_dict['batch_size'],
        #                         shuffle=True,
        #                         **kwargs)
        src_loader = DataLoader(IndexSampler(num_src_node),
                                batch_size=num_src_node,
                                shuffle=True,
                                **kwargs)
        tar_loader = DataLoader(IndexSampler(num_tar_node),
                                batch_size=num_tar_node,
                                shuffle=True,
                                **kwargs)
        time_st = time.time()
        for epoch in range(hyperpara_dict['epochs']):
            gw = 0
            trans_tmp = np.zeros(self.trans.shape)
            if scheduler is not None:
                scheduler.step()

            for src_idx, indices1 in enumerate(src_loader):
                for tar_idx, indices2 in enumerate(tar_loader):
                    # Estimate Gromov-Wasserstein discrepancy give current costs
                    # cost_s, cost_t, mu_s, mu_t, index_s, index_t, mask_s, mask_t = \
                    #     cost_sampler2(database, indices1, indices2, device)

                    # if hyperpara_dict['display']:
                    #     self.plot_result(index_s, index_t, epoch, prefix=hyperpara_dict['prefix'])


                    cost1 = database['cost_s'].cuda()
                    cost2 = database['cost_t'].cuda()
                    mu_s= (torch.ones(cost1.shape[1],1)/cost1.shape[1]).cuda()
                    mu_t= (torch.ones(cost2.shape[1],1)/cost2.shape[1]).cuda()
                    cost12 = 0

                    trans, d_gw, cost_12 = self.regularized_gromov_wasserstein_discrepancy(cost1, cost2, cost12,
                                                                                           mu_s, mu_t, hyperpara_dict)
                    # estimate optimal transport
                    # trans_np = trans.cpu().data.numpy()
                    # index_s_np = index_s.cpu().data.numpy()
                    # index_t_np = index_t.cpu().data.numpy()
                    # patch = self.trans[index_s_np, :]
                    # patch = patch[:, index_t_np]
                    # energy = np.sum(patch) + 1
                    # for row in range(trans_np.shape[0]):
                    #     for col in range(trans_np.shape[1]):
                    #         trans_tmp[index_s_np[row], index_t_np[col]] += (energy * trans_np[row, col])

                    # gw += d_gw

                    # if epoch == 0:
                    #     sgd_iter = hyperpara_dict['sgd_iteration']
                    # else:
                    #     sgd_iter = 100

                    # inner iteration based on SGD
                    # for num in range(sgd_iter):
                    #     # zero the parameter gradients
                    #     optimizer.zero_grad()
                    #     # Update source and target embeddings alternatively
                    #     loss_gw, loss_w, regularizer = self.gwl_model(index_s, index_t, trans,
                    #                                                   mu_s, mu_t, cost_s, cost_t,
                    #                                                   prior=cost_12, mask1=mask_s,
                    #                                                   mask2=mask_t, mask12=None)
                    #     loss = 1e3 * loss_gw + 1e3 * loss_w + regularizer
                    #     loss.backward()
                    #     optimizer.step()
                    #     if num % 10 == 0:
                    #         print('inner {}/{}: loss={:.6f}.'.format(num, sgd_iter, loss.data))

                    a1, a5, a10, a30 =my_check_align(trans.T,database['ground_truth'])
                    # self.NC1.append(nc1)
                    # self.NC2.append(nc2)
                    # self.EC1.append(ec1)
                    # self.EC2.append(ec2)

                    # logger.info('Train Epoch: {}'.format(epoch))
                    # # logger.info('- node correctness: {:.4f}%, {:.4f}%'.format(nc1, nc2))
                    # # logger.info('- edge correctness: {:.4f}%, {:.4f}%'.format(ec1, ec2))
                    # logger.info('- node correctness: {:.4f}%, {:.4f}%'.format(nc1, nc2))
                    
                    
                    # print('{} Edge Noise:{} Feat Noise:{} Type:{} GW beta:{}  ep:{}'.format(
                    #         hyperpara_dict['dataset'], hyperpara_dict['edge_noise'], hyperpara_dict['feat_noise'], hyperpara_dict['noise_type'],hyperpara_dict['gw_beta'], hyperpara_dict['epoch']))
                    # print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@30 %.2f%% Time: %.2fs' % (a1 * 100, a5 * 100, a10 * 100, a30 * 100, time_cost))
                    # with open('result.txt', 'a+') as f:
                    #     f.write('{} Edge Noise:{} Feat Noise:{} Type:{} Bases:{} GW beta:{} ss:{}, ep:{}\n'.format(
                    #         args.dataset, args.edge_noise, args.feat_noise, args.noise_type, args.bases, args.gw_beta, args.step_size, args.epoch))
                    #     f.write('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@30 %.2f%% Time: %.2fs\n ' % (a1 * 100, a5 * 100, a10 * 100, a30 * 100, time_cost))
                    # break
                # if src_idx % 100 == 1:
                #     logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                #         epoch, src_idx * hyperpara_dict['batch_size'],
                #         len(src_loader.dataset), 100. * src_idx / len(src_loader)))
                # break    
            # logger.info('- GW distance = {:.4f}.'.format(gw/len(src_loader)))

            trans_tmp /= np.max(trans_tmp)
            self.trans = trans_tmp
            self.d_gw.append(gw/len(src_loader))
            
            time_ed = time.time()
            time_cost = time_ed - time_st
            print('{} Edge Noise:{} Feat Noise:{} Type:{} GW beta:{}  ep:{}'.format(
                    hyperpara_dict['dataset'], hyperpara_dict['edge_noise'], hyperpara_dict['feat_noise'], hyperpara_dict['noise_type'],hyperpara_dict['beta'], hyperpara_dict['epochs']))
            print('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@30 %.2f%% Time: %.2fs' % (a1 * 100, a5 * 100, a10 * 100, a30 * 100, time_cost))
            with open('result.txt', 'a+') as f:
                f.write('{} Edge Noise:{} Feat Noise:{} Type:{} GW beta:{}  ep:{}\n'.format(
                    hyperpara_dict['dataset'], hyperpara_dict['edge_noise'], hyperpara_dict['feat_noise'], hyperpara_dict['noise_type'],hyperpara_dict['beta'], hyperpara_dict['epochs']))
                f.write('H@1 %.2f%% H@5 %.2f%% H@10 %.2f%% H@30 %.2f%% Time: %.2fs\n ' % (a1 * 100, a5 * 100, a10 * 100, a30 * 100, time_cost))
            # break


    def obtain_embedding(self, hyperpara_dict, index, idx):
        device = torch.device('cuda:0' if hyperpara_dict['use_cuda'] else 'cpu')
        self.gwl_model.to(device)
        self.gwl_model.eval()
        return self.gwl_model.emb_model[idx](index)

    def save_model(self, full_path, mode: str = 'entire'):
        """
        Save trained model
        :param full_path: the path of directory
        :param mode: 'parameter' for saving only parameters of the model,
                     'entire' for saving entire model
        """
        if mode == 'entire':
            torch.save(self.gwl_model, full_path)
            logger.info('The entire model is saved in {}.'.format(full_path))
        elif mode == 'parameter':
            torch.save(self.gwl_model.state_dict(), full_path)
            logger.info('The parameters of the model is saved in {}.'.format(full_path))
        else:
            logger.warning("'{}' is a undefined mode, we use 'entire' mode instead.".format(mode))
            torch.save(self.gwl_model, full_path)
            logger.info('The entire model is saved in {}.'.format(full_path))

    def load_model(self, full_path, mode: str = 'entire'):
        """
        Load pre-trained model
        :param full_path: the path of directory
        :param mode: 'parameter' for saving only parameters of the model,
                     'entire' for saving entire model
        """
        if mode == 'entire':
            self.gwl_model = torch.load(full_path)
        elif mode == 'parameter':
            self.gwl_model.load_state_dict(torch.load(full_path))
        else:
            logger.warning("'{}' is a undefined mode, we use 'entire' mode instead.".format(mode))
            self.gwl_model = torch.load(full_path)

    def save_matching(self, full_path):
        with open(full_path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.NC1, self.EC1, self.NC2, self.EC2, self.d_gw], f)

    def save_recommend(self, full_path):
        with open(full_path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([self.Prec, self.Recall, self.F1, self.d_gw, self.trans], f)

