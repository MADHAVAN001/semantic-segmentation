import dataloader.coco
import models.unet
import yaml 
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import scipy.misc
import os
import math

fp = open("configs/coco_unet.yaml")
cfg = yaml.load(fp)

data_gen_train = dataloader.coco.CocoDataGenerator(cfg, "train")
data_gen_valid = dataloader.coco.CocoDataGenerator(cfg, "validate")

classes = dataloader.coco.get_super_class(cfg)
num_classes = len(classes)

epochs, nfilters, dropout, kernel_size, batch_norm = dataloader.coco.get_model_hyperparams(cfg)

img_width, img_height, n_chan = data_gen_train.image_width, data_gen_train.image_height, data_gen_train.image_num_chans

wt_path = dataloader.coco.get_model_check_path(cfg)

num_train  = data_gen_train.num_images
num_valid =  data_gen_valid.num_images
batch_size = data_gen_train.batch_size

steps_per_epoch_train = num_train/batch_size
steps_per_epoch_val = num_valid/batch_size

unet_inst = models.unet.uNetModel(img_width, img_height, n_chan, num_classes, nfilters, kernel_size, dropout, batch_norm)

unet_inst.model.compile(optimizer = Adam(), loss = "sparse_categorical_crossentropy",metrics=["sparse_categorical_accuracy"])

train = False
inference = True
eval_validation = False
use_test = True
num_inference = 20
path_infer_store = "./predict/"
scale_factor = math.floor((255.0 / num_classes))

if train:
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(wt_path, verbose=1, save_best_only=True, save_weights_only=True)
    ]

    results = unet_inst.model.fit_generator(data_gen_train, initial_epoch=0, verbose = 1, steps_per_epoch = steps_per_epoch_train, 
                                            validation_data = data_gen_valid, callbacks = callbacks, 
                                            epochs = epochs, validation_steps = steps_per_epoch_val)

if inference:
    unet_inst.model.load_weights(wt_path)
    if use_test:
        dataset_use = data_gen_train
    else:
        dataset_use = data_gen_valid
    
    path_fetch = dataset_use.labels_dir
    path_fetch = path_fetch  + "/"

    files = os.listdir(path_fetch)
    files = files[0:num_inference]

    if eval_validation:
        unet_inst.model.evaluate(data_gen_valid, verbose = 1)

    predictions = unet_inst.model.predict(dataset_use, verbose = 1)
    predictions = np.argmax(predictions, -1)

    for x in range(num_inference):
        true_mask = np.load(path_fetch+files[x])
        prediction_inst = predictions[x,:,:] * scale_factor
        true_mask = true_mask * scale_factor
        scipy.misc.imsave(path_infer_store+files[x].replace(".jpg.npy","_pred.jpg"), prediction_inst)
        scipy.misc.imsave(path_infer_store+files[x].replace(".jpg.npy","_true.jpg"), true_mask)
