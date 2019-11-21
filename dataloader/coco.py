import itertools
import os

import cv2
import numpy as np
from keras.utils import Sequence

import utils.prefixer
from utils.file_utils import file_line_count


class CocoDataGenerator(Sequence):

    def __init__(self, cfg, run_type):
        fetch_prefix = utils.prefixer.fetch_prefix(run_type)

        self.batch_size = cfg[fetch_prefix]["batch_size"]
        self.index_file = cfg["data"][fetch_prefix]["index_file"]
        self.dataset_dir = cfg["data"][fetch_prefix]["dataset_dir"]
        self.labels_dir = cfg["data"][fetch_prefix]["labels_dir"]

    def __len__(self):
        return int(np.ceil(file_line_count(self.index_file)/float(self.batch_size)))

    def __getitem__(self, idx):

        batch_images = list()
        batch_labels = list()

        with open(self.index_file) as f:
            result = itertools.islice(f, idx*self.batch_size, (idx+1)*self.batch_size)

            for line in result:
                batch_images.append(cv2.imread(os.path.join(self.dataset_dir, line.strip())))
                batch_labels.append(np.load(os.path.join(self.dataset_dir, line.strip()+".npy")))

        return np.array(batch_images), np.array(batch_labels)
