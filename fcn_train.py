import yaml
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

import dataloader.coco
import models.fcn


def load_coco_dataset(cfg):
    training_data_generator = dataloader.coco.CocoDataGenerator(cfg, "train")
    validation_data_generator = dataloader.coco.CocoDataGenerator(cfg, "validate")
    classes = dataloader.coco.get_super_class(cfg)

    return training_data_generator, validation_data_generator, classes


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

    fcn = models.fcn.FCN()

    model = fcn.resnet50(input_shape=(img_width, img_height, channels_count), classes=classes_count)

    model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True)
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
