import os
import sys
import time
import random
import pickle
import numpy as np
import argparse
from os.path import isfile, join
from sklearn.model_selection import train_test_split

sys.path.insert(0, './src/utils')
import data_utils as du

"""
Script for creating dataset to predict. Separate sets with different galaxies. 

[GPS - 03/09/2019]
"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Create candidate slices')
    parser.add_argument('--slice-size', type=int, default=22,
                        help='window size for visualization (slice size: sz x sz)')
    parser.add_argument('--testing', type=bool, default=True,
                        help='Set the mode for Testing')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    sz = args.slice_size

    dataset_info = 'raw_'
   
    dirm = 'data/'
    targets_txt = 'targets.txt'
    tabs_txt = 'tab_links.txt'

    # dir with created slices of galaxy targets:
    data_dir = dirm+dataset_info+str(sz)+'x'+str(sz)

    tin = time.time()
    #files = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
    #files = sorted(files)
    files = []
    with open(targets_txt) as file:
        for target in file:
            files.append(target[:-1] + '.dat')

    print('creating dataset...')
    data = np.array([], dtype=np.int64).reshape(0,5,sz,sz)
    coords = np.array([], dtype=np.int64).reshape(0,2)
    ids = np.array([], dtype=np.int64).reshape(0,)
    galaxies = np.array([], dtype=np.int64).reshape(0,)
    labels = np.array([], dtype=np.int64).reshape(0,)
    for i in range(len(files)):
        t_ini = time.time()
        file_name = join(data_dir,files[i])
        if not os.path.exists(file_name): sys.exit('ERROR: file %s does not exist'%file_name)
        target_data, target_coords, target_ids, target_classes = du.load_db(file_name)
        data = np.concatenate((data, target_data), axis=0)
        coords = np.concatenate((coords, target_coords), axis=0)
        ids = np.concatenate((ids, target_ids), axis=0)
        labels = np.concatenate((classes, target_classes), axis=0)
        strs = [files[i][:-4] for x in range(len(target_ids))]
        galaxies = np.concatenate((galaxies, strs), axis=0)
    
    # Save test set
    if args.testing:     

        db_name = 'test_'+dataset_info+str(sz)+'x'+str(sz)
        dataset = {'data':data, 'coordinates':coords, 'galaxies':galaxies, 'ids':ids, 'classes': labels}
        with open(os.path.join('data', db_name)+'.dat', 'wb') as outfile:
                        pickle.dump(dataset, outfile, pickle.HIGHEST_PROTOCOL)
        print('dataset shape: %s' % (str(data.shape)))

    else:
        data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        test_db_name = 'test_'+dataset_info+str(sz)+'x'+str(sz)
        train_db_name = 'train_'+dataset_info+str(sz)+'x'+str(sz)
        train_dataset = {'data':data_train, 'labels': label_train}
        test_dataset = {'data':data_test, 'labels': label_test}
        with open(os.path.join('data', train_db_name)+'.dat', 'wb') as outfile:
                pickle.dump(train_dataset, outfile, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join('data', test_db_name)+'.dat', 'wb') as outfile:
                pickle.dump(test_dataset, outfile, pickle.HIGHEST_PROTOCOL)
          
