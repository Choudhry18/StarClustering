import sys
import os
import pickle
import argparse
import numpy as np
import torch
from torchvision import transforms

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
    print(f"Shape of the data: {data[0].shape}")
    transformed_data = []
    transformed_labels = []

    # Loop over each item in the dataset
    for i, (datum, label) in enumerate(zip(data, labels)):
        if label == 0:
            # Perform horizontal and vertical flips for label 0
            h_flip = np.fliplr(datum)
            v_flip = np.flipud(datum)
            transformed_data.extend([h_flip, v_flip])
            transformed_labels.extend([label, label])
        elif label in [1, 2]:
            # Perform horizontal flip for labels 1 and 2
            h_flip = np.fliplr(datum)
            transformed_data.append(h_flip)
            transformed_labels.append(label)

    # Append transformed data and labels to the original dataset
    data = np.concatenate((data, np.array(transformed_data)))
    labels = np.concatenate((labels, np.array(transformed_labels)))

    # Verify the label frequencies in the updated dataset
    label_frequencies = np.bincount(labels)
    for label, frequency in enumerate(label_frequencies):
        print(f"Label {label}: {frequency} occurrences")
    datum_shape = data[0].shape
    print(f"Shape of the first datum: {datum_shape}")
# Data augmentation transforms
# data_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     RandomScale(scale_factor=1.07, probability=0.5),
#     transforms.RandomRotation([90, 270])
# ])
