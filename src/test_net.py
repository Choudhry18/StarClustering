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

from starcnet import Net

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a model for star cluser classification')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
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
                        help='checkpoint to load network', default='best_model.pth', type=str)
    args = parser.parse_args()
    return args


args = parse_args()

with open('data/test_raw_32x32.dat', 'rb') as infile:
    dset = pickle.load(infile)
data, label = dset['data'], dset['labels']
mean = np.load(args.data_dir+'mean.npy')

# subtract mean
data -= mean[np.newaxis,:,np.newaxis,np.newaxis]

tdata = torch.from_numpy(data)
tdata = tdata.float()
tlabel = torch.from_numpy(np.transpose(label))
tlabel = tlabel.long()
test = torch_du.TensorDataset(tdata, tlabel)
test_loader = torch_du.DataLoader(test, batch_size=args.test_batch_size, shuffle=False)


def eval(test_loader, args):
    model.eval()
    correct = 0
    num_data = 0
    # predictions = np.array([], dtype=np.int64).reshape(0)
    # scores = np.array([], dtype=np.float32).reshape(0,4)
    # targets = np.array([], dtype=np.int64).reshape(0)
    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum().item()
            num_data += len(data)
            # predictions = np.concatenate((predictions, pred.cpu().numpy()))
            # targets = np.concatenate((targets, target.data.cpu().numpy()))
            # scores = np.concatenate((scores, output.data.cpu().numpy()), axis=0)
    # Calculate accuracy after the loop
    test_accuracy = 100. * correct / num_data
    return test_accuracy
    # return acccuracy_batch, targets, predictions, scores


if __name__ == '__main__':

    args = parse_args()
    args.dataset = args.dataset + "_raw_32x32.dat"
    dataset_path = os.path.join(args.data_dir, args.dataset)
    with open(dataset_path, 'rb') as infile:
        dset = pickle.load(infile)
    data, label = dset['data'], dset['labels']
    print('dataset shape: %s' % (str(data.shape)))
    print('label shape: %s' % (str(label.shape)))
    mean = np.load(args.data_dir+'mean.npy')

    data -= mean[np.newaxis,:,np.newaxis,np.newaxis]

    tdata = torch.from_numpy(data)
    tdata = tdata.float()
    tlabel = torch.from_numpy(np.transpose(label))
    tlabel = tlabel.long()
    test = torch_du.TensorDataset(tdata, tlabel)
    test_loader = torch_du.DataLoader(test, batch_size=args.test_batch_size, shuffle=False)
    
    args.cuda = args.cuda and torch.cuda.is_available()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = Net()
    
    if args.checkpoint != '':
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.save_dir+args.checkpoint, map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size() }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.cuda:
        model.cuda()

    start_time = time.time()
    test_accuracy = eval(test_loader, args)     
    # save scores (predictions + targets)
    print(test_accuracy)
