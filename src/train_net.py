#! /usr/bin/env python

import os
import sys
import time
import numpy as np

sys.path.insert(0, './src/utils')
sys.path.insert(0, './model')

import data_utils as du
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_du
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from starcnet import Net


def train(train_loader, model, criterion, optimizer, args):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return loss.item()


if __name__ == '__main__':

    args = parse_args()

    # loading dataset
    data_test, _, _, true_label = du.load_db(os.path.join(args.data_dir,'test_'+args.dataset+'.dat'))
    label_test = np.zeros((data_test.shape[0]))
    mean = np.load(args.data_dir+'mean.npy')

    # subtract mean
    data_test -= mean[np.newaxis,:,np.newaxis,np.newaxis]

    tdata = torch.from_numpy(data_test)
    tdata = tdata.float()
    tlabel = torch.from_numpy(np.transpose(label_test))
    tlabel = tlabel.long()
    testd = torch_du.TensorDataset(tdata, tlabel)
    test_loader = torch_du.DataLoader(testd, batch_size=args.test_batch_size, shuffle=False) 
    
    args.cuda = args.cuda and torch.cuda.is_available()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = Net()
    
    if args.checkpoint != '':
        model_dict = model.state_dict()
        if args.cuda:
            pretrained_dict = torch.load(args.save_dir+args.checkpoint)
        else:
            pretrained_dict = torch.load(args.save_dir+args.checkpoint, map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size() }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.cuda:
        model.cuda()

    start_time = time.time()
    test_accuracy, targets, predictions, scores = test(test_loader, args)     
    # save scores (predictions + targets)
    np.save(os.path.join('output','scores'), scores)
