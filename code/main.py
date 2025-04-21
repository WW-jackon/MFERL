import torch
import random
import numpy as np
from datapro import loading_data

from train import train_test


# import warnings
#
# warnings.filterwarnings("ignore")


class Config2:
    def __init__(self):
        self.datapath = './datasets/821-2115-9589'
        self.kfold = 5
        self.batchSize = 128
        # self.ratio = 0.2
        self.epoch = 30

        # self.gcn_layers = 2
        # self.view = 3
        self.fm = 128
        self.fd = 128
        self.inSize = 128


        self.outSize = 128
        # self.nodeNum = 0
        self.hdnDropout = 0.5
        self.fcDropout = 0.5
        # self.maskMDI = False

        self.channel_num = 5
        self.embedding_dim = 128

        self.miRNA_numbers = 821
        self.circRNA_numbers = 2115

        self.ssl_temp = 0.1
        self.proto_reg = 8e-8
        self.ssl_reg = 1e-5
        self.alpha = 1

        self.r = 0.5

        self.doc_dim = 256
        self.role = 256
        self.pro_dim = 128

        self.ctd_dim = 30
        self.doc2vec_dim = 256
        self.role2vec_dim = 256
        self.kmer_dim = 340
        self.lr =0.01

        self.device = torch.device('cuda')


def main():
    param = Config2()
    train_data = loading_data(param)
    result = train_test(train_data, param, state='valid')



if __name__ == "__main__":
    main()
