import sys
import os
import pickle
import argparse
import numpy as np

sys.path.insert(0, './src/utils')
import data_utils as du
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Create candidate slices')
    parser.add_argument('--dataset', type=str, default="run",
                        help='The dataset to augment')
    parser.add_argument('--data_dir', type=str, default="data/",
                    help='The dataset to augment')
    args = parser.parse_args()
    return args

# Custom scaling transformation
class RandomScale:
    def __init__(self, scale_factor=1.07, probability=0.5):
        self.scale_factor = scale_factor
        self.probability = probability

    def __call__(self, x):
        if np.random.rand() < self.probability:
            scale = self.scale_factor
            size = (int(x.shape[1] * scale), int(x.shape[2] * scale))
            x = transforms.Resize(size)(x)
            x = transforms.CenterCrop((32, 32))(x)  # Assuming original size is 32x32
        return x
if __name__ == '__main__':
    
    args = parse_args()
    args.dataset = args.dataset + "_raw_32x32.dat"
    dataset_path = os.path.join(args.data_dir, args.dataset)
    with open(dataset_path, 'rb') as infile:
        dset = pickle.load(infile)
    data, labels = dset['data'] , dset['labels']
    labels = labels.astype(int)
    label_frequencies = np.bincount(labels)
    for label, frequency in enumerate(label_frequencies):
        print(f"Label {label}: {frequency} occurrences")

# Data augmentation transforms
# data_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     RandomScale(scale_factor=1.07, probability=0.5),
#     transforms.RandomRotation([90, 270])
# ])
