import numpy as np
import cv2
import math
import sys
import time

sys.path.append("..")

import dataloader.coco
import models.fcn_sparsify
from tensorflow_model_optimization.sparsity import keras as sparsity


def load_config_params(cfg):
    set_sparse = cfg["training"]["sparsify"]

    sparsify_params = list()
    sparsify_params.append(cfg["training"]["initial_sparse"])
    sparsify_params.append(cfg["training"]["final_sparse"])
    sparsify_params.append(cfg["training"]["initial_sparse_step"])
    sparsify_params.append(cfg["training"]["final_sparse_step"])
    sparsify_params.append(cfg["training"]["sparse_freq"])

    return set_sparse, sparsify_params


def load_coco_dataset(cfg):
    validation_data_generator = dataloader.coco.CocoDataGenerator(cfg, "validate")
    classes = dataloader.coco.get_super_class(cfg)

    return validation_data_generator, classes


def load_fcn_model(cfg, checkpoint_path, input_shape, output_classes_count):
    is_sparse_enabled, sparsify_params = load_config_params(cfg)

    fcn = models.fcn_sparsify.FCN(is_sparse_enabled, sparsify_params)
    model = fcn.resnet50(input_shape=input_shape, classes=output_classes_count)
    model = strip_sparsified_model(is_sparse_enabled, model)
    model.load_weights(checkpoint_path)

    return model


def measure_validation_time(model, validation_data_generator):
    start = time.time()
    model.evaluate(validation_data_generator, verbose=1)
    end = time.time()
    return end - start


def strip_sparsified_model(is_sparsify_enabled, model):
    if is_sparsify_enabled:
        return sparsity.strip_pruning(model)

    return model


def analyize_prediction(data_generator, classes_count, model, input_file_name):
    scale_factor = math.floor((255.0 / classes_count))

    path_fetch_output = data_generator.labels_dir
    path_fetch_output = path_fetch_output + "/"
    path_fetch_input = data_generator.dataset_dir + "/"

    files_output = input_file_name.replace(".jpg", ".jpg.npy")
    true_mask = np.load(path_fetch_output + files_output)
    true_image = cv2.imread(path_fetch_input + input_file_name)
    # Reshape to a tensor
    true_image_pass = true_image.reshape(1, true_image.shape[0], true_image.shape[1], true_image.shape[2])
    prediction = model.predict(true_image_pass, verbose=1)
    prediction = np.argmax(prediction, -1)
    prediction = prediction[0, :, :]

    prediction = prediction * scale_factor
    true_mask = true_mask * scale_factor

    return true_image, true_mask, prediction
