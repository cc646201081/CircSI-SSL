import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


def Trainer(data_type, ratio, epochs, model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, test_dl, device,
            logger, experiment_log_dir, training_mode):
    # Start training and testing
    logger.debug("Training and Testing started ....")

    criterion = nn.NLLLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    max_test_auc = 0
    max_test_acc = 0
    max_test_precision = 0
    max_test_recall = 0
    max_epoch = 0

    for epoch in range(1, epochs + 1):
        # Train and validate
        train_loss, train_acc, train_auc, train_precision, train_recall = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, device, training_mode)
        test_loss, test_acc, test_auc, test_precision, test_recall, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)

        if max_test_auc <= test_auc:
            max_epoch = epoch
            max_test_acc = test_acc
            max_test_auc = test_auc
            max_test_precision = test_precision
            max_test_recall = test_recall

        scheduler.step(train_loss)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'Train Loss : {train_loss:.4f}\t | \t Accuracy : {train_acc:2.4f} | \t AUC : {train_auc:2.4f} | \t Precision : {train_precision:2.4f} | \t Recall : {train_recall:2.4f}\n'
                         f'Test Loss : {test_loss:.4f}\t | \t Accuracy : {test_acc:2.4f} | \t AUC : {test_auc:2.4f} | \t Precision : {test_precision:2.4f} | \t Recall : {test_recall:2.4f}')
        else:
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'Train Loss : {train_loss:.4f}\t')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        logger.debug(f'\nepoch:{max_epoch}\t | \tmax_test_acc: {max_test_acc:2.4f} \t | \tmax_test_auc: {max_test_auc:2.4f} | \tmax_test_precision: {max_test_precision:2.4f} | \tmax_test_recall: {max_test_recall:2.4f}\n')
        with open("experiments_logs/result_all.txt",'a',encoding='utf-8') as f :
            f.write("dataset:{} {}:{}\n".format(data_type,ratio,10-ratio))
            f.write(f'epoch:{max_epoch}\t | \tmax_test_acc: {max_test_acc:2.4f} \t | \tmax_test_auc: {max_test_auc:2.4f} | \tmax_test_precision: {max_test_precision:2.4f} | \tmax_test_recall: {max_test_recall:2.4f}\n')

    logger.debug("\n################## Training and Testing is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, device, training_mode):
    total_loss = []
    total_acc = []
    total_auc = []
    total_precision = []
    total_recall = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data1, data2, data3, labels) in enumerate(train_loader):
        # send to device
        data1, data2, data3 = data1.float().to(device), data2.float().to(device), data3.float().to(device)
        labels = labels.long().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        features1 = model(data1,tag=1)
        features2 = model(data2,tag=2)
        features3 = model(data3,tag=3)

        if training_mode == "self_supervised":
            temp_cont_loss23 = temporal_contr_model(features1, features2, features3)
            temp_cont_loss32 = temporal_contr_model(features1, features3, features2)
            loss = temp_cont_loss23 + temp_cont_loss32
        else:
            yt = temporal_contr_model(features1, features2, features3, 1)
            loss = criterion(yt, labels)

            auc = roc_auc_score(labels.cpu(), yt.detach().cpu()[:,1])
            acc = accuracy_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
            precision = precision_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
            recall = recall_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))

            total_acc.append(acc)
            total_precision.append(precision)
            total_recall.append(recall)
            total_auc.append (auc)

        total_loss.append(loss.item())
        loss.backward()

        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
        total_auc = 0
        total_precision = 0
        total_recall = 0
        return total_loss, total_acc, total_auc, total_precision, total_recall
    else:
        total_acc = torch.tensor(total_acc).mean()
        total_auc = torch.tensor(total_auc).mean()
        total_precision = torch.tensor(total_precision).mean()
        total_recall = torch.tensor(total_recall).mean()

        return total_loss, total_acc, total_auc, total_precision, total_recall


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_precision = []
    total_recall = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data1, data2, data3, labels in test_dl:
            data1, data2, data3 = data1.float().to(device), data2.float().to(device), data3.float().to(device)
            labels = labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                features1 = model(data1, tag=1)
                features2 = model(data2, tag=2)
                features3 = model(data3, tag=3)

                yt = temporal_contr_model(features1, features2, features3, 1)

            # compute loss
            if training_mode != "self_supervised":
                loss = criterion(yt, labels)
                auc = roc_auc_score(labels.cpu(), yt.detach().cpu()[:,1])
                acc = accuracy_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
                precision = precision_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
                recall = recall_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))

                total_loss.append(loss.item())
                total_acc.append(acc)
                total_precision.append(precision)
                total_recall.append(recall)
                total_auc.append (auc)

                pred = yt.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        total_auc = torch.tensor(total_auc).mean()  # average acc
        total_precision = torch.tensor(total_precision).mean()
        total_recall = torch.tensor(total_recall).mean()
        return total_loss, total_acc, total_auc, total_precision, total_recall, outs, trgs
    else:
        total_loss = 0
        total_acc = 0
        total_auc = 0
        total_precision = 0
        total_recall = 0
        return total_loss, total_acc, total_auc, total_precision, total_recall, [], []
