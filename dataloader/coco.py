import itertools
import os

import cv2
import numpy as np
from keras.utils import Sequence

import utils.prefixer
from utils.file_utils import file_line_count
import json

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
                batch_labels.append(np.load(os.path.join(self.labels_dir, line.strip()+".npy")))

        return np.array(batch_images), np.array(batch_labels)
    
    def get_super_class(self, cfg):
        path = cfg["data"]["classes"]["index_file"]
        json_file = open(path)
        data = json.load(json_file)
        data = data['categories']
        supercat_names = {}
        supercat_ids = {}
        class_superclass_map_name = {}
        class_superclass_map_id = {}
        lencat = len(data)
        for x in range(lencat):
            supercat = data[x]['supercategory']
            if not supercat in supercat_names.keys():
                supercat_names[supercat] = []
                supercat_ids[supercat] = []
            supercat_names[supercat].append(data[x]['name'])
            supercat_ids[supercat].append(data[x]['id'])
            class_superclass_map_name[data[x]['name']] = supercat
            class_superclass_map_id[data[x]['id']] = supercat
        final_list_supercat = list(supercat_names.keys())
        final_list_supercat.sort()
        idxBG = final_list_supercat.index('background')
        tmp = final_list_supercat[idxBG]
        final_list_supercat[idxBG] = final_list_supercat[0]
        final_list_supercat[0] = tmp
        #Override for now due to bug in code to generate mask. Once fixed, below line can be removed
        #TODO: Run regeneration of mask and then remove below line 
        final_list_supercat = ['background', 'appliance', 'electronic', 'accessory', 'kitchen', 'sports', 'vehicle', 'furniture', 'food', 'outdoor', 'indoor', 'animal', 'person']
        return final_list_supercat


