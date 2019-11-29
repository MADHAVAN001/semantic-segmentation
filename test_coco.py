import dataloader.coco
import models.unet
import yaml 
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import numpy as np
import scipy.misc
import os
import math
import cv2

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
use_test = False
start_inference = 0
end_inference = 50
path_infer_store = "./predict/"
scale_factor = math.floor((255.0 / num_classes))
enable_debug = 0

if train:
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(wt_path, verbose=1, save_best_only=True, save_weights_only=True),
        CSVLogger(cfg["training"]["csv_logger_path"])
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
    
    path_fetch_output = dataset_use.labels_dir
    path_fetch_output = path_fetch_output  + "/"
    path_fetch_input = dataset_use.dataset_dir + "/"

    files_input = os.listdir(path_fetch_input)
    files_input = files_input[start_inference:end_inference]

    if eval_validation:
        unet_inst.model.evaluate(data_gen_valid, verbose = 1)

    for x in range(start_inference,end_inference,1):
        files_output = files_input[x].replace(".jpg",".jpg.npy")
        true_mask = np.load(path_fetch_output+files_output)
        true_image = cv2.imread(path_fetch_input+files_input[x])
        #Reshape to a tensor
        true_image_pass = true_image.reshape(1,true_image.shape[0],true_image.shape[1],true_image.shape[2])
        prediction = unet_inst.model.predict(true_image_pass, verbose = 1)
        prediction = np.argmax(prediction, -1)
        prediction = prediction[0,:,:]
        if enable_debug:
            print("------")
            print(files_input[x])
            print(np.unique(prediction))
            print(np.unique(true_mask))
            print("------")
        prediction = prediction * scale_factor
        true_mask = true_mask * scale_factor
        scipy.misc.imsave(path_infer_store+files_input[x], true_image)
        scipy.misc.imsave(path_infer_store+files_input[x].replace(".jpg","_pred.jpg"), prediction)
        scipy.misc.imsave(path_infer_store+files_output.replace(".jpg.npy","_true.jpg"), true_mask)
