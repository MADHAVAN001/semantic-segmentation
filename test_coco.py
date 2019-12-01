import dataloader.coco
import models.unet_tf
import yaml 
from utils.performance import PerformanceMetrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import numpy as np
import scipy.misc
import os
import math
import cv2
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow as tf
import tempfile
import zipfile

fp = open("configs/coco_unet.yaml")
cfg = yaml.load(fp)

data_gen_train = dataloader.coco.CocoDataGenerator(cfg, "train")
data_gen_valid = dataloader.coco.CocoDataGenerator(cfg, "validate")

classes, class_names = dataloader.coco.get_super_class(cfg)
num_classes = len(classes)

#TODO: Add condition here to see if to categorize by superclass or class
if True:
    num_classes = len(class_names)

epochs, nfilters, dropout, kernel_size, batch_norm, sparsify, sparsify_params = dataloader.coco.get_model_hyperparams(cfg)

img_width, img_height, n_chan = data_gen_train.image_width, data_gen_train.image_height, data_gen_train.image_num_chans

wt_path = dataloader.coco.get_model_check_path(cfg)

num_train  = data_gen_train.num_images
num_valid =  data_gen_valid.num_images
batch_size = data_gen_train.batch_size

steps_per_epoch_train = num_train/batch_size
steps_per_epoch_val = num_valid/batch_size

unet_inst = models.unet_tf.uNetModel(img_width, img_height, n_chan, num_classes, nfilters, kernel_size, dropout, 
                                    batch_norm, sparsify = sparsify, sparsify_params = sparsify_params)

unet_inst.model.compile(optimizer = Adam(), loss = "sparse_categorical_crossentropy",metrics=["sparse_categorical_accuracy"])
unet_inst.model.summary()

train = True
inference = False
eval_validation = False
use_test = False
start_inference = 50
end_inference = 100
path_infer_store = "./predict/"
pruning_summaries_dir = "./workspace/"
scale_factor = math.floor((255.0 / num_classes))
full_model_path = wt_path.replace(".h5","_full.h5")
full_striped_model_path = wt_path.replace(".h5","_full_stripped_sparse.h5")
size_compare = False
quantize = False
quantize_path = wt_path.replace(".h5","_full_stripped_sparse_quantized.h5")
enable_debug = 0

model_path = wt_path + "unet_weights-{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.h5"
if train:
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(wt_path, verbose=1, save_best_only=True, save_weights_only=True),
        sparsity.UpdatePruningStep(),
        CSVLogger(cfg["csv_logger_path"]),
        PerformanceMetrics(cfg["performance_logger_path"]),
        sparsity.PruningSummaries(log_dir=pruning_summaries_dir)
    ]

    results = unet_inst.model.fit_generator(data_gen_train, initial_epoch=0, verbose = 1, steps_per_epoch = steps_per_epoch_train, 
                                            validation_data = data_gen_valid, callbacks = callbacks, 
                                            epochs = epochs, validation_steps = steps_per_epoch_val)
unet_inst.model.load(weights(wt_path))
tf.keras.models.save_model(unet_inst.model, full_model_path, include_optimizer=False)

if inference:
    unet_inst.model.load_weights(wt_path)
    if sparsify:
        unet_inst.model = sparsity.strip_pruning(unet_inst.model)
        tf.keras.models.save_model(unet_inst.model,full_striped_model_path, include_optimizer=False)
    
    if size_compare == True:
        _, zip2 = tempfile.mkstemp('.zip')
        with zipfile.ZipFile(zip2, 'w', compression=zipfile.ZIP_DEFLATED) as f:
             f.write(full_striped_model_path)
        print("Size of the pruned model before compression: %.2f Mb" %
                (os.path.getsize(full_striped_model_path) / float(2**20)))
        print("Size of the pruned model after compression: %.2f Mb" % 
                (os.path.getsize(zip2) / float(2**20)))
    if quantize: 
        converter = tf.lite.TFLiteConverter.from_keras_model(unet_inst.model)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_quant_model = converter.convert()
        tflite_quant_model_file = './pretrained/sem_seg_unet_2019_27_11_82_6_pruned_tflite_quant.h5'
        with open(quantize_path, 'wb') as f:
            f.write(tflite_quant_model)
    
        img_og, label_og = data_gen_valid.__getitem__(0)
        input_tensor[:,:,:] = img_og[0,:,:,:]
        tmp = img_og[0,:,:,:] 
        tmp = tmp.astype("float32")
        tmp = tmp.reshape((1,tmp.shape[0],tmp.shape[1],tmp.shape[2]))
        interpreter = tf.lite.Interpreter(model_path=str(tflite_quant_model_file))
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        interpreter.set_tensor(input_index, tmp)

        interpreter.invoke()
        output_index = interpreter.get_output_details()[0]["index"]
        output = interpreter.get_tensor(output_index)

        print(type(interpreter))
    
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
