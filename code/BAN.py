import torch
from torch.nn.utils.weight_norm import weight_norm
from torch import nn
import torch.nn.functional as F




class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.4, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        # print('v_num:', v_num)
        # print('v.shape:', v.shape)
        q_num = q.size(1)
        # print('q_num:', q_num)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            # print('v_shape:', v_.shape)
            q_ = self.q_net(q)
            # print('q_shape:', q_.shape)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            # print('v_.shape2:', self.v_net(v).shape)
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)  # [1, 384, 962, 1]
            # print('v_.shape:', v_.shape)

            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)  # [1, 384, 1, 2346]
            # print('q_.shape:', q_.shape)

            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q # [1, 384, 962, 2346]
            # print('d_.shape:', d_.shape)

            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out  [1, 962, 2346, 128]
            # print('att_maps.shape:', att_maps.shape)
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q  [1, 128, 962, 2346]
            # print('att_maps2.shape:', att_maps.shape)
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        return att_maps


class FCNet(nn.Module):
   

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class MLP_Projector(nn.Module):
    def __init__(self, in_dim, map_nums, out_dim):
        super(MLP_Projector, self).__init__()
        self.fc1 = nn.Linear(in_dim * map_nums, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.out(x)
        return x

