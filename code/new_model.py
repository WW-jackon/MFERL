import pandas as pd
import numpy as np
from torch import nn as nn
import torch
import os
from torch_geometric.nn import GCNConv
from mcam import mcam
from BAN import BANLayer, MLP_Projector
import torch.nn.functional as F



def pro_data(data, em, ed):
    edgeData = data.t()

    mFeaData = em
    dFeaData = ed
    m_index = edgeData[0]
    d_index = edgeData[1]
    Em = torch.index_select(mFeaData, 0, m_index)
    Ed = torch.index_select(dFeaData, 0, d_index)

    return Em, Ed


class MLP(nn.Module):
    def __init__(self, inSize, outSize, dropout, actFunc, outBn=True, outAct=False, outDp=False):
        super(MLP, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        # self.out = nn.Linear(inSize + inSize + inSize, outSize)
        self.out = nn.Linear(inSize*3, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        x = self.out(x)  # batchsize*featuresize
        if self.outBn: x = self.bns(x) if len(x.shape) == 2 else self.bns(x.transpose(-1, -2)).transpose(-1, -2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x


# model control
class MSFEICL(nn.Module):
    def __init__(self, args):
        super(MSFEICL, self).__init__()
        self.args = args
        self.mc_graph_dataset = dict()
        # miRNA feature
        self.miRNA_ctd = pd.read_csv(os.path.join(self.args.datapath + '/miRNA_ctd_dict.txt'), header=None).iloc[:,
                         1:].values
        # print(self.miRNA_ctd.shape)
        self.miRNA_doc2vec = pd.read_csv(os.path.join(self.args.datapath + '/miRNA_doc2vec_dict.txt'),
                                         header=None).iloc[:,
                             1:].values
        self.miRNA_kmer = pd.read_csv(os.path.join(self.args.datapath + '/miRNA_kmer_dict.txt'), header=None).iloc[
                          :, 1:].values
        self.miRNA_role2vec = pd.read_csv(os.path.join(self.args.datapath + '/miRNA_role2vec_dict.txt'),
                                          header=None).iloc[:,
                              1:].values
        self.miRNA_squence = np.loadtxt(os.path.join(self.args.datapath + '/miRNA_sequence_similarity.csv'),
                                        delimiter=',')
        # miRNA simi graph
        self.miRNA_simlarity_graph = torch.LongTensor(np.argwhere(self.miRNA_squence > self.args.r).T)
        self.mc_graph_dataset['mm_ctd'] = {'data_matrix': self.miRNA_ctd, 'edges': self.miRNA_simlarity_graph}
        self.mc_graph_dataset['mm_kmer'] = {'data_matrix': self.miRNA_kmer, 'edges': self.miRNA_simlarity_graph}
        self.mc_graph_dataset['mm_doc'] = {'data_matrix': self.miRNA_doc2vec, 'edges': self.miRNA_simlarity_graph}
        self.mc_graph_dataset['mm_role'] = {'data_matrix': self.miRNA_role2vec, 'edges': self.miRNA_simlarity_graph}

        self.gcn_x_m_ctd = GCNConv(self.args.ctd_dim, self.args.ctd_dim).cuda()
        self.gcn_x_m_kmer = GCNConv(self.args.kmer_dim, self.args.kmer_dim).cuda()
        self.gcn_x_m_doc = GCNConv(self.args.doc_dim, self.args.doc_dim).cuda()
        self.gcn_x_m_role = GCNConv(self.args.role, self.args.role).cuda()

        self.gcn_x_m_ctd_p = nn.Linear(self.args.ctd_dim, self.args.pro_dim).cuda()
        self.gcn_x_m_kmer_p = nn.Linear(self.args.kmer_dim, self.args.pro_dim).cuda()
        self.gcn_x_m_doc_p = nn.Linear(self.args.doc_dim, self.args.pro_dim).cuda()
        self.gcn_x_m_role_p = nn.Linear(self.args.role, self.args.pro_dim).cuda()
        # self.gcn_x_m_ctd = GCNConv(882,882).cuda()
        # self.gcn_x_m_ctd_p = nn.Linear(882, self.args.pro_dim).cuda()

        self.cnn_m = nn.Conv2d(in_channels=self.args.channel_num, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)

        # circRNA feature
        self.circRNA_ctd = pd.read_csv(os.path.join(self.args.datapath + '/circRNA_ctd_dict.txt'), header=None).iloc[:,
                           1:].values
        self.circRNA_doc2vec = pd.read_csv(os.path.join(self.args.datapath + '/circRNA_doc2vec_dict.txt'),
                                           header=None).iloc[:,
                               1:].values
        self.circRNA_kmer = pd.read_csv(os.path.join(self.args.datapath + '/circRNA_kmer_dict.txt'), header=None).iloc[
                            :, 1:].values
        self.circRNA_role2vec = pd.read_csv(os.path.join(self.args.datapath + '/circRNA_role2vec_dict.txt'),
                                            header=None).iloc[:,
                                1:].values
        self.circRNA_squence = np.loadtxt(os.path.join(self.args.datapath + '/circRNA_sequence_similarity.csv'),
                                          delimiter=',')
        # circRNA simi graph
        self.circRNA_simlarity_graph = torch.LongTensor(np.argwhere(self.circRNA_squence > self.args.r).T)
        self.mc_graph_dataset['cc_ctd'] = {'data_matrix': self.circRNA_ctd, 'edges': self.circRNA_simlarity_graph}
        self.mc_graph_dataset['cc_kmer'] = {'data_matrix': self.circRNA_kmer, 'edges': self.circRNA_simlarity_graph}
        self.mc_graph_dataset['cc_doc'] = {'data_matrix': self.circRNA_doc2vec, 'edges': self.circRNA_simlarity_graph}
        self.mc_graph_dataset['cc_role'] = {'data_matrix': self.circRNA_role2vec, 'edges': self.circRNA_simlarity_graph}

        self.gcn_x_c_ctd = GCNConv(self.args.ctd_dim, self.args.ctd_dim).cuda()
        self.gcn_x_c_kmer = GCNConv(self.args.kmer_dim, self.args.kmer_dim).cuda()
        self.gcn_x_c_doc = GCNConv(self.args.doc_dim, self.args.doc_dim).cuda()
        self.gcn_x_c_role = GCNConv(self.args.role, self.args.role).cuda()
        # self.gcn_x_c_ctd = GCNConv(882, 882).cuda()

        self.gcn_x_c_ctd_p = nn.Linear(self.args.ctd_dim, self.args.pro_dim).cuda()
        self.gcn_x_c_kmer_p = nn.Linear(self.args.kmer_dim, self.args.pro_dim).cuda()
        self.gcn_x_c_doc_p = nn.Linear(self.args.doc_dim, self.args.pro_dim).cuda()
        self.gcn_x_c_role_p = nn.Linear(self.args.role, self.args.pro_dim).cuda()
        # self.gcn_x_c_ctd_p = nn.Linear(882, self.args.pro_dim).cuda()

        self.cnn_c = nn.Conv2d(in_channels=self.args.channel_num, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)

        self.dropout = nn.Dropout(p=self.args.fcDropout)
        self.sigmoid = nn.Sigmoid()
        self.cbam_m = cbam(self.args.channel_num)
        self.cbam_c = cbam(self.args.channel_num)

        # feature project for miRNA
        self.pj_m_ctd = nn.Linear(self.args.ctd_dim, self.args.embedding_dim).cuda()
        self.pj_m_doc2vec = nn.Linear(self.args.doc2vec_dim, self.args.embedding_dim).cuda()
        self.pj_m_kmer = nn.Linear(self.args.kmer_dim, self.args.embedding_dim).cuda()
        self.pj_m_role2vec = nn.Linear(self.args.role2vec_dim, self.args.embedding_dim).cuda()
        self.pj_m_squence = nn.Linear(self.args.miRNA_numbers, self.args.embedding_dim).cuda()

        # feature project for circRNA
        self.pj_c_ctd = nn.Linear(self.args.ctd_dim, self.args.embedding_dim).cuda()
        self.pj_c_doc2vec = nn.Linear(self.args.doc2vec_dim, self.args.embedding_dim).cuda()
        self.pj_c_kmer = nn.Linear(self.args.kmer_dim, self.args.embedding_dim).cuda()
        self.pj_c_role2vec = nn.Linear(self.args.role2vec_dim, self.args.embedding_dim).cuda()
        self.pj_c_squence = nn.Linear(self.args.circRNA_numbers, self.args.embedding_dim).cuda()

        #############################################
        # self.fc1_m = nn.Linear(self.args.embedding_dim * self.args.channel_num, self.args.embedding_dim).cuda()
        self.fc1_m = nn.Linear(self.args.embedding_dim * 4, self.args.embedding_dim).cuda()
        self.bn1_m = nn.BatchNorm1d(self.args.embedding_dim)
        self.fc2_m = nn.Linear(self.args.embedding_dim, self.args.embedding_dim).cuda()
        self.bn2_m = nn.BatchNorm1d(self.args.embedding_dim)
        self.out_m = nn.Sigmoid()

        # self.fc1_d = nn.Linear(self.args.embedding_dim * self.args.channel_num, self.args.embedding_dim).cuda()
        self.fc1_d = nn.Linear(self.args.embedding_dim * 4, self.args.embedding_dim).cuda()
        self.bn1_d = nn.BatchNorm1d(self.args.embedding_dim)
        self.fc2_d = nn.Linear(self.args.embedding_dim, self.args.embedding_dim).cuda()
        self.bn2_d = nn.BatchNorm1d(self.args.embedding_dim)
        self.out_d = nn.Sigmoid()

        #############################################

        self.fc_ban_m = nn.Linear(self.args.embedding_dim * self.args.channel_num, self.args.embedding_dim).cuda()
        self.fc_ban_c = nn.Linear(self.args.embedding_dim * self.args.channel_num, self.args.embedding_dim).cuda()

        self.embedding_dim = self.args.embedding_dim

        self.bcn = BANLayer(v_dim=self.embedding_dim, q_dim=self.embedding_dim, h_dim=self.embedding_dim,
                            h_out=self.embedding_dim)

        self.miRNA_projector = MLP_Projector(self.args.circRNA_numbers, self.embedding_dim, self.embedding_dim)
        self.circRNA_projector = MLP_Projector(self.args.miRNA_numbers, self.embedding_dim, self.embedding_dim)

        self.relu1 = nn.ReLU()
        self.fcLinear = MLP(self.args.outSize, 1, dropout=self.args.fcDropout, actFunc=self.relu1).cuda()

    def miRNA_embeding(self, miRNA_ctd, miRNA_doc2vec, miRNA_kmer, miRNA_role2vec, miRNA_squence):
        miRNA_ctd = torch.from_numpy(miRNA_ctd).float().cuda()
        miRNA_doc2vec = torch.from_numpy(miRNA_doc2vec).float().cuda()
        miRNA_role2vec = torch.from_numpy(miRNA_role2vec).float().cuda()
        miRNA_kmer = torch.from_numpy(miRNA_kmer).float().cuda()
        miRNA_squence = torch.from_numpy(miRNA_squence).float().cuda()

        # #change homo
        # data = self.mc_graph_dataset
        # XM_raw = torch.cat((miRNA_ctd, miRNA_kmer, miRNA_doc2vec, miRNA_role2vec), 1)
        # x_m = torch.relu(
        #     self.gcn_x_m_ctd(XM_raw.float().cuda(),data['mm_ctd']['edges'].cuda()))
        # x_m = self.dropout(self.gcn_x_m_ctd_p(x_m))
        # gcn
        data = self.mc_graph_dataset
        x_m_f1 = torch.relu(
            # self.gcn_x_m_ctd(self.x_m_ctd.cuda(), data['mm_ctd']['edges'].cuda()))
            self.gcn_x_m_ctd(torch.from_numpy(data['mm_ctd']['data_matrix']).float().cuda(),
                             data['mm_ctd']['edges'].cuda()))
        x_m_f1_p = self.dropout(self.gcn_x_m_ctd_p(x_m_f1))
        # print('过了')

        x_m_f2 = torch.relu(
            self.gcn_x_m_kmer(torch.from_numpy(data['mm_kmer']['data_matrix']).float().cuda(),
                              data['mm_kmer']['edges'].cuda()))
        x_m_f2_p = self.dropout(self.gcn_x_m_kmer_p(x_m_f2))

        x_m_f3 = torch.relu(
            self.gcn_x_m_doc(torch.from_numpy(data['mm_doc']['data_matrix']).float().cuda(),
                             data['mm_doc']['edges'].cuda()))
        x_m_f3_p = self.dropout(self.gcn_x_m_doc_p(x_m_f3))

        x_m_f4 = torch.relu(
            self.gcn_x_m_role(torch.from_numpy(data['mm_role']['data_matrix']).float().cuda(),
                              data['mm_role']['edges'].cuda()))
        x_m_f4_p = self.dropout(self.gcn_x_m_role_p(x_m_f4))

        m_ctd = self.dropout(self.pj_m_ctd(miRNA_ctd))
        m_doc2vec = self.dropout(self.pj_m_doc2vec(miRNA_doc2vec))
        m_role2vec = self.dropout(self.pj_m_role2vec(miRNA_role2vec))
        m_kmer = self.dropout(self.pj_m_kmer(miRNA_kmer))
        m_squence = self.dropout(self.pj_m_squence(miRNA_squence))

        XM_raw = torch.cat((m_ctd, m_doc2vec, m_role2vec, m_kmer, m_squence), 1).t()  # [640,962]
        XM_raw_p = self.fc_ban_m(XM_raw.t())
        x_m = torch.cat((x_m_f1_p, x_m_f2_p, x_m_f3_p, x_m_f4_p), 1)  # [962,640]

        XM = XM_raw.view(1, self.args.channel_num, self.args.fm, -1)

        # add 特征的通道和空间注意力
        XM_channel_attention = self.cbam_m(XM)  # [1,5,128,962]
        XM_channel_attention = torch.relu(XM_channel_attention)

        x = self.cnn_m(XM_channel_attention)
        x = x.view(self.args.fm, self.args.miRNA_numbers).t()
        # XM_raw.t()
        # return x, XM_raw.t()

        return x, x_m, XM_raw_p#CBAM的结果【128维】、GCN的结果【640维】、原始的拼接降维【128维】
        # return x_m,XM_raw_p
    def circRNA_embeding(self, circRNA_ctd, circRNA_doc2vec, circRNA_kmer, circRNA_role2vec, circRNA_squence):
        circRNA_ctd = torch.from_numpy(circRNA_ctd).float().cuda()
        circRNA_doc2vec = torch.from_numpy(circRNA_doc2vec).float().cuda()
        circRNA_role2vec = torch.from_numpy(circRNA_role2vec).float().cuda()
        circRNA_kmer = torch.from_numpy(circRNA_kmer).float().cuda()
        circRNA_squence = torch.from_numpy(circRNA_squence).float().cuda()

        # change homo
        # data = self.mc_graph_dataset
        # XC_raw = torch.cat((circRNA_ctd, circRNA_kmer, circRNA_doc2vec, circRNA_role2vec), 1)
        # x_c = torch.relu(
        #     self.gcn_x_c_ctd(XC_raw.float().cuda(), data['cc_ctd']['edges'].cuda()))
        # x_c = self.dropout(self.gcn_x_c_ctd_p(x_c))
        # gcn
        data = self.mc_graph_dataset
        x_c_f1 = torch.relu(
            # self.gcn_x_c_ctd(self.x_c_ctd.cuda(), data['cc_ctd']['edges'].cuda()))
            self.gcn_x_c_ctd(torch.from_numpy(data['cc_ctd']['data_matrix']).float().cuda(),
                             data['cc_ctd']['edges'].cuda()))
        x_c_f1_p = self.dropout(self.gcn_x_c_ctd_p(x_c_f1))

        x_c_f2 = torch.relu(
            self.gcn_x_c_kmer(torch.from_numpy(data['cc_kmer']['data_matrix']).float().cuda(),
                              data['cc_kmer']['edges'].cuda()))
        x_c_f2_p = self.dropout(self.gcn_x_c_kmer_p(x_c_f2))

        x_c_f3 = torch.relu(
            self.gcn_x_c_doc(torch.from_numpy(data['cc_doc']['data_matrix']).float().cuda(),
                             data['cc_doc']['edges'].cuda()))
        x_c_f3_p = self.dropout(self.gcn_x_c_doc_p(x_c_f3))

        x_c_f4 = torch.relu(
            self.gcn_x_c_role(torch.from_numpy(data['cc_role']['data_matrix']).float().cuda(),
                              data['cc_role']['edges'].cuda()))
        x_c_f4_p = self.dropout(self.gcn_x_c_role_p(x_c_f4))

        c_ctd = self.dropout(self.pj_c_ctd(circRNA_ctd))
        c_doc2vec = self.dropout(self.pj_c_doc2vec(circRNA_doc2vec))
        c_role2vec = self.dropout(self.pj_c_role2vec(circRNA_role2vec))
        c_kmer = self.dropout(self.pj_c_kmer(circRNA_kmer))
        c_squence = self.dropout(self.pj_c_squence(circRNA_squence))

        XC_raw = torch.cat((c_ctd, c_doc2vec, c_role2vec, c_kmer, c_squence), 1).t()
        XC_raw_p = self.fc_ban_c(XC_raw.t())

        x_c = torch.cat((x_c_f1_p, x_c_f2_p, x_c_f3_p, x_c_f4_p), 1)

        XC = XC_raw.view(1, self.args.channel_num, self.args.fd, -1)


        XC_channel_attention = self.cbam_m(XC)  # [1,5,128,2346]
        XC_channel_attention = torch.relu(XC_channel_attention)

        y = self.cnn_m(XC_channel_attention)  # [1,1,128,2346]
        y = y.view(self.args.fd, self.args.circRNA_numbers).t()

        return y, x_c, XC_raw_p
        # return x_c,  XC_raw_p


    def ssl_layer_loss(self, context_miRNA_emb_all, context_circRNA_emb_all, initial_miRNA_emb_all,
                       initial_circRNA_emb_all, miRNA, circRNA):
        context_miRNA_emb = context_miRNA_emb_all[miRNA]  # [2048,64]
        initial_miRNA_emb = initial_miRNA_emb_all[miRNA]  # [2048,64]
        norm_miRNA_emb1 = F.normalize(context_miRNA_emb)  # [2048,64]
        norm_miRNA_emb2 = F.normalize(initial_miRNA_emb)  # [2048,64]
        norm_all_miRNA_emb = F.normalize(initial_miRNA_emb_all)  # [410,64]
        pos_score_miRNA = torch.mul(norm_miRNA_emb1, norm_miRNA_emb2).sum(dim=1)  # [2048]
        ttl_score_miRNA = torch.matmul(norm_miRNA_emb1, norm_all_miRNA_emb.transpose(0, 1))  # [2048,410]
        pos_score_miRNA = torch.exp(pos_score_miRNA / self.args.ssl_temp)  # 一个值
        ttl_score_miRNA = torch.exp(ttl_score_miRNA / self.args.ssl_temp).sum(dim=1)  # 一个值
        ssl_loss_miRNA = -torch.log(pos_score_miRNA / ttl_score_miRNA).sum()  # 一个值

        context_circRNA_emb = context_circRNA_emb_all[circRNA]  # [2048,64]
        initial_circRNA_emb = initial_circRNA_emb_all[circRNA]  # [2048,64]
        norm_circRNA_emb1 = F.normalize(context_circRNA_emb)  # [2048,64]
        norm_circRNA_emb2 = F.normalize(initial_circRNA_emb)  # [2048,64]
        norm_all_circRNA_emb = F.normalize(initial_circRNA_emb_all)  # [1931,64]
        pos_score_circRNA = torch.mul(norm_circRNA_emb1, norm_circRNA_emb2).sum(dim=1)  # [2048]
        ttl_score_circRNA = torch.matmul(norm_circRNA_emb1, norm_all_circRNA_emb.transpose(0, 1))  # [2048,1931]
        pos_score_circRNA = torch.exp(pos_score_circRNA / self.args.ssl_temp)  # [2048]
        ttl_score_circRNA = torch.exp(ttl_score_circRNA / self.args.ssl_temp).sum(dim=1)  # [2048]
        ssl_loss_circRNA = -torch.log(pos_score_circRNA / ttl_score_circRNA).sum()  # 一个值

        ssl_loss = self.args.ssl_reg * (ssl_loss_miRNA + self.args.alpha * ssl_loss_circRNA)  # 一个值
        return ssl_loss

    def forward(self, train_data):  ##*

        Em, Em_cat, XM_raw_p = self.miRNA_embeding(self.miRNA_ctd, self.miRNA_doc2vec, self.miRNA_kmer,
                                                   self.miRNA_role2vec,
                                                   self.miRNA_squence)
        Ed, Ed_cat, XC_raw_p = self.circRNA_embeding(self.circRNA_ctd, self.circRNA_doc2vec, self.circRNA_kmer,
                                                     self.circRNA_role2vec, self.circRNA_squence)


        Em_cat_project = self.out_m(self.bn2_m(self.fc2_m(self.bn1_m(self.fc1_m(Em_cat)))))
        Ec_cat_project = self.out_d(self.bn2_d(self.fc2_d(self.bn1_d(self.fc1_d(Ed_cat)))))


        re_Em = XM_raw_p.unsqueeze(0)
        re_Ec = XC_raw_p.unsqueeze(0)

        att = self.bcn(re_Em, re_Ec, softmax=True)  # [1, 128, 962, 2346]

        att = att.permute(0, 2, 3, 1).squeeze(0)  # [962, 2346, 128]

        m_nums, c_nums, map_nums = att.shape  # [962, 2346, 128]

        att_m, att_c = att.reshape(m_nums, -1), att.reshape(c_nums, -1)  # att_m:[962,300288]  att_c:[2346,123136]


        XM_raw_bcn_project = self.miRNA_projector(att_m)  # [962,128]
        # self.Em_bcn_project

        XC_raw_bcn_project = self.circRNA_projector(att_c)  # [2346,128]

        edgeData = train_data.t()
        m_index = edgeData[0]
        c_index = edgeData[1]

        #
        # ssl_loss_1 = self.ssl_layer_loss(Em, Ed, Em_cat_project, Ec_cat_project, m_index,
        #                                  c_index)
        #
        # ssl_loss_2 = self.ssl_layer_loss(XM_raw_bcn_project, XC_raw_bcn_project, Em_cat_project, Ec_cat_project,
        #                                  m_index,
        #                                  c_index)
        #
        ssl_loss_1 = self.ssl_layer_loss(Em_cat_project, Ec_cat_project,XM_raw_p,XC_raw_p,m_index,c_index)
        ssl_loss_2 = self.ssl_layer_loss(Em, Ed, XM_raw_p,XC_raw_p, m_index, c_index)
        ssl_loss_3 = self.ssl_layer_loss(XM_raw_bcn_project, XC_raw_bcn_project, XM_raw_p,XC_raw_p, m_index, c_index)

        final_miRNA_fea = torch.cat((Em, Em_cat_project, XM_raw_bcn_project), dim=1)
        final_circRNA_fea = torch.cat((Ed, Ec_cat_project, XC_raw_bcn_project), dim=1)


        mFea, dFea = pro_data(train_data, final_miRNA_fea, final_circRNA_fea)

        node_embed = (mFea.unsqueeze(1) * dFea.unsqueeze(1)).squeeze(dim=1)
        pre_part = self.fcLinear(node_embed)

        pre_asso = self.sigmoid(pre_part).squeeze(dim=1)

        ssl_loss = ssl_loss_1 + ssl_loss_2 +ssl_loss_3
        # ssl_loss = 0.0

        return pre_asso, ssl_loss

