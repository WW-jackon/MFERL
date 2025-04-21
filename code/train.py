import time
import torch
from datapro import CVEdgeDataset
from new_model import MSFEICL
import numpy as np
from sklearn import metrics
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
from clac_metric import get_metric
import csv
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def get_metrics(score, label):
    y_pre = score
    y_true = label
    # metric = caculate_metrics(y_pre, y_true)
    metric = get_metric(y_true, y_pre)
    return metric

def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'f1_score ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'precision ：%.4f ' % (list[5]),
          'specificity ：%.4f \n' % (list[6]))


def train_test(train_data, param, state):
    valid_metric = []
    valid_tpr = []
    valid_fpr = []
    valid_recall = []
    valid_precision = []
    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    kfolds = param.kfold
    torch.manual_seed(42)

    if state == 'valid':
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
        train_idx, valid_idx = [], []
        for train_index, valid_index in kf.split(train_edges):
            train_idx.append(train_index)
            valid_idx.append(valid_index)
        for i in range(kfolds):
            model = MSFEICL(param)  ##*
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=0.0)  ###

            print(f'Fold {i + 1} ')
            # get train set and valid set
            edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
            labels_train, labels_valid = train_labels[train_idx[i]], train_labels[valid_idx[i]]
            trainEdges = CVEdgeDataset(edges_train, labels_train)
            validEdges = CVEdgeDataset(edges_valid, labels_valid)
            trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)
            validLoader = DataLoader.DataLoader(validEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)

            print("-----training-----")
            for e in range(param.epoch):
                running_loss = 0.0  ###
                epo_label = []
                epo_score = []
                # print("epoch：", e + 1)
                model.train()
                start = time.time()
                for i, item in enumerate(trainLoader):
                    data, label = item
                    train_data = data.cuda()
                    true_label = label.cuda()  ###
                    pre_score, ssl_loss = model(train_data)  ##*
                    train_loss = torch.nn.BCELoss()
                    loss = train_loss(pre_score, true_label) + ssl_loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()  ###
                    # print(f"After batch {i + 1}: loss= {loss:.3f};", end='\n')  ###
                    batch_score = pre_score.cpu().detach().numpy()
                    epo_score = np.append(epo_score, batch_score)
                    epo_label = np.append(epo_label, label.numpy())
                end = time.time()
                # print('Time：%.2f \n' % (end - start))

            valid_score, valid_label = [], []  ###
            model.eval()
            with torch.no_grad():
                # print("-----validing-----")
                for i, item in enumerate(validLoader):
                    data, label = item
                    train_data = data.cuda()
                    pre_score, loss = model(train_data)  ##*
                    batch_score = pre_score.cpu().detach().numpy()
                    valid_score = np.append(valid_score, batch_score)
                    valid_label = np.append(valid_label, label.numpy())
                # end = time.time()
                # print('Time：%.2f \n' % (end - start))
                tpr, fpr, recall_list, precision_list,metric = get_metrics(valid_score, valid_label)
                print_met(metric)
                valid_metric.append(metric)
                valid_tpr.append(tpr)
                valid_fpr.append(fpr)
                valid_recall.append(recall_list)
                valid_precision.append(precision_list)
        # print(np.array(valid_metric))
        # 格式化输出每个数值
        formatted_valid_metric = [[round(item, 4) for item in sublist] for sublist in valid_metric]
        # 打印结果
        for sublist in formatted_valid_metric:
            print(sublist)
        cv_metric = np.mean(valid_metric, axis=0)
        print_met(cv_metric)
        cv_tpr = np.mean(valid_tpr, axis=0)
        cv_fpr = np.mean(valid_fpr, axis=0)
        cv_recall = np.mean(valid_recall, axis=0)
        cv_precision = np.mean(valid_precision, axis=0)

        valid_tpr.append(cv_tpr)
        valid_fpr.append(cv_fpr)
        valid_recall.append(cv_recall)
        valid_precision.append(cv_precision)

        StorFile(valid_tpr, '5cv/tpr_all_ours_9589.csv')
        StorFile(valid_fpr, '5cv/fpr_all_ours_9589.csv')
        StorFile(valid_recall, '5cv/recall_all_ours_9589.csv')
        StorFile(valid_precision, '5cv/precision_all_ours_9589.csv')

    return kfolds
