#! /usr/bin/env python

import os
import sys
import time
import numpy as np
import pickle
sys.path.insert(0, './src/utils')
sys.path.insert(0, './model')

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_du
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from starcnet import Net
import torch.nn.init as init


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a model for star cluser classification')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--data_dir', dest='data_dir', help='test dataset directory',
                        default='data/', type=str)
    parser.add_argument('--dataset', dest='dataset', help='training dataset file reference',
                        default='raw_32x32', type=str)
    parser.add_argument('--gpu', dest='gpu', help='CUDA visible device',
                        default='', type=str)
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_dir', dest='save_dir', help='save dir for scores',
                        default='model/', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network', default='', type=str)
    parser.add_argument('--name', dest='name',
                        help='trained model name', default='best_model', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # loading dataset
    with open('data/train_raw_32x32.dat', 'rb') as infile:
        dset = pickle.load(infile)
    data, label = dset['data'], dset['labels']
    mean = np.load(args.data_dir+'mean.npy')
    label_counts = np.bincount(label)
    
    data -= mean[np.newaxis,:,np.newaxis,np.newaxis]

    train_data, val_data, train_labels, val_labels = train_test_split(data, label, test_size=0.1, random_state=42)
    with open('data/ngc3274_raw_32x32.dat', 'rb') as infile:
        dset = pickle.load(infile)
    data, label = dset['data'], dset['labels']
    data -= mean[np.newaxis,:,np.newaxis,np.newaxis]
    val_data = np.concatenate((val_data, data), axis=0)
    val_labels = np.concatenate((val_labels, label), axis=0)
    
    tdata = torch.from_numpy(train_data)
    tdata = tdata.float()
    tlabel = torch.from_numpy(train_labels)
    tlabel = tlabel.long()
    vdata = torch.from_numpy(val_data)
    vdata = vdata.float()
    vlabel = torch.from_numpy(val_labels)
    vlabel = vlabel.long()
    print(tdata.shape, tlabel.shape)
    print(vdata.shape, vlabel.shape)
    testd = torch_du.TensorDataset(tdata, tlabel)
    train_loader = torch_du.DataLoader(testd, batch_size=args.test_batch_size, shuffle=False) 
    vald = torch_du.TensorDataset(vdata, vlabel)
    val_loader = torch_du.DataLoader(testd, batch_size=args.test_batch_size, shuffle=False) 
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = Net()
    # ADAM Optimizer
    model.apply(init_weights)
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Gradient clipping, something to change using swarm optmization
    max_grad_norm = 1.0  # Adjust this value if needed
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    if args.checkpoint != '':
        model_dict = model.state_dict()
        if args.cuda:
            pretrained_dict = torch.load(args.save_dir+args.checkpoint + '.pth')
        else:
            pretrained_dict = torch.load(args.save_dir+args.checkpoint, map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size() }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.cuda:
        model.cuda()


    start_time = time.time()
    # Training loop
    num_epochs = 30
    best_val_loss = float('inf')
    patience = 3  # for early stopping
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
    
        for data, target in train_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            # Assuming 'output' is your model's output tensor and 'target' is the target tensor
            if torch.isnan(target).any():
                print("NaN values found in output or target tensors")
            if torch.isinf(output).any() or torch.isinf(target).any():
                print("inf values found in output or target tensors")
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    # Save the trained model
    torch.save(model.state_dict(), args.save_dir + args.name + '.pth')