import itertools
import os

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

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
        self.image_width = cfg["data"]["dimensions"]["img_width"]
        self.image_height = cfg["data"]["dimensions"]["img_height"]
        self.image_num_chans = cfg["data"]["dimensions"]["img_num_chans"]
        self.num_images = file_line_count(self.index_file)

    def __len__(self):
        return int(np.ceil(self.num_images/float(self.batch_size)))

    def __getitem__(self, idx):

        batch_images = list()
        batch_labels = list()

        with open(self.index_file) as f:
            result = itertools.islice(f, idx*self.batch_size, (idx+1)*self.batch_size)

            for line in result:
                batch_images.append(cv2.imread(os.path.join(self.dataset_dir, line.strip())))
                label_mat= np.load(os.path.join(self.labels_dir, line.strip()+".npy"))
                label_mat = label_mat.reshape(label_mat.shape[0],label_mat.shape[1],1)
                batch_labels.append(label_mat)

        return np.array(batch_images), np.array(batch_labels)

def get_super_class(cfg):
    path = cfg["data"]["classes"]["index_file"]
    json_file = open(path)
    data = json.load(json_file)
    data = data['categories']
    supercat_names = {}
    supercat_ids = {}
    class_superclass_map_name = {}
    class_superclass_map_id = {}
    lencat = len(data)
    class_name = []
    for x in range(lencat):
        supercat = data[x]['supercategory']
        class_name.append(data[x]['name'])
        if not supercat in supercat_names.keys():
            supercat_names[supercat] = []
            supercat_ids[supercat] = []
        supercat_names[supercat].append(data[x]['name'])
        supercat_ids[supercat].append(data[x]['id'])
        class_superclass_map_name[data[x]['name']] = supercat
        class_superclass_map_id[data[x]['id']] = supercat
    final_list_supercat = list(supercat_names.keys())
    final_list_supercat.sort()
    idx_bg = final_list_supercat.index('background')
    tmp = final_list_supercat[idx_bg]
    final_list_supercat[idx_bg] = final_list_supercat[0]
    final_list_supercat[0] = tmp
    #Override for now due to bug in code to generate mask. Once fixed, below line can be removed
    #TODO: Run regeneration of mask and then remove below line
    final_list_supercat = ['background', 'appliance', 'electronic', 'accessory', 'kitchen', 'sports', 'vehicle', 'furniture', 'food', 'outdoor', 'indoor', 'animal', 'person']
    return final_list_supercat, class_name

def get_model_hyperparams(cfg):
    
    n_epochs = cfg["training"]["num_epochs"]
    n_filters = cfg["training"]["num_filters"]
    dropout = cfg["training"]["dropout"]
    kernel_size = cfg["training"]["kernel_size"]
    batch_norm = cfg["training"]["batch_norm"]
    set_sparse = cfg["training"]["sparsify"]
    sparsify_params = []
    sparsify_params.append(cfg["training"]["initial_sparse"]
    sparsify_params.append(cfg["training"]["final_sparse"])
    sparsify_params.append(cfg["training"]["initial_sparse_step"])
    sparsify_params.append(cfg["training"]["final_sparse_step"])
    sparsify_params.append(cfg["training"]["sparse_freq"])
    return n_epochs, n_filters, dropout, kernel_size, batch_norm, set_sparse, sparsify_params

def get_model_check_path(cfg):
    return cfg["training"]["model_check_path"]
