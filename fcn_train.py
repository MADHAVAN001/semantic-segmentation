import yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam

import dataloader.coco
from utils.performance import PerformanceMetrics
import models.fcn_sparsify
from tensorflow_model_optimization.sparsity import keras as sparsity


def load_coco_dataset(cfg):
    training_data_generator = dataloader.coco.CocoDataGenerator(cfg, "train")
    validation_data_generator = dataloader.coco.CocoDataGenerator(cfg, "validate")
    classes = dataloader.coco.get_super_class(cfg)

    return training_data_generator, validation_data_generator, classes


def load_config_params(cfg):
    set_sparse = cfg["training"]["sparsify"]

    sparsify_params = list()
    sparsify_params.append(cfg["training"]["initial_sparse"])
    sparsify_params.append(cfg["training"]["final_sparse"])
    sparsify_params.append(cfg["training"]["initial_sparse_step"])
    sparsify_params.append(cfg["training"]["final_sparse_step"])
    sparsify_params.append(cfg["training"]["sparse_freq"])

    return set_sparse, sparsify_params


def main():
    fp = open("configs/coco_fcn.yaml")
    cfg = yaml.load(fp)

    training_data_generator, validation_data_generator, classes = load_coco_dataset(cfg)

    classes_count = len(classes)

    epochs, filters_count, dropout, kernel_size, batch_norm = dataloader.coco.get_model_hyperparams(cfg)

    img_width, img_height = training_data_generator.image_width, training_data_generator.image_height
    channels_count = training_data_generator.image_num_chans

    # Get the path for saving checkpoints
    checkpoint_path = dataloader.coco.get_model_check_path(cfg)

    train_samples_count = training_data_generator.num_images
    validation_samples_count = validation_data_generator.num_images
    batch_size = training_data_generator.batch_size

    steps_per_epoch_train = train_samples_count / batch_size
    steps_per_epoch_val = validation_samples_count / batch_size

    is_sparse_enabled, sparse_configs = load_config_params(cfg)

    fcn = models.fcn_sparsify.FCN(is_sparse_enabled, sparse_configs)

    model = fcn.resnet50(input_shape=(img_width, img_height, channels_count), classes=classes_count)

    model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    model_path = checkpoint_path + "fcn_weights-{epoch:02d}-{val_acc:.2f}.h5"
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=True),
        CSVLogger(cfg["csv_logger_path"]),
        PerformanceMetrics(cfg["performance_logger_path"]),
        sparsity.UpdatePruningStep(),
        sparsity.PruningSummaries(log_dir="/workspace/")
    ]

    results = model.fit_generator(
        training_data_generator,
        initial_epoch=0,
        verbose=1,
        steps_per_epoch=steps_per_epoch_train,
        validation_data=validation_data_generator,
        callbacks=callbacks,
        epochs=epochs,
        validation_steps=steps_per_epoch_val
    )


if __name__ == "__main__":
    main()
