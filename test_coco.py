import dataloader.coco
import models.unet
import yaml 
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

fp = open("configs/coco_unet.yaml")
cfg = yaml.load(fp)

data_gen_train = dataloader.coco.CocoDataGenerator(cfg, "train")
data_gen_valid = dataloader.coco.CocoDataGenerator(cfg, "validate")

classes = dataloader.coco.get_super_class(cfg)
numClasses = len(classes)

img_width, img_height, n_chan = 128, 128, 3
nfilters = 16
kernel_size = 3
dropout = 0.05
batchNorm = True
batch_size = 40
epochs = 50

numTrain  = 117266
numValid =  4952

steps_per_epoch_train = numTrain/batch_size
steps_per_epoch_val = numValid/batch_size

unet_inst = models.unet.uNetModel(img_width, img_height, n_chan, numClasses, nfilters, kernel_size, dropout, batchNorm)

unet_inst.model.compile(optimizer = Adam(), loss = "mean_squared_error",metrics=["accuracy"])

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-semseg.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = unet_inst.model.fit_generator(data_gen_train, initial_epoch=0, verbose = 1, steps_per_epoch = steps_per_epoch_train, 
                                        validation_data = data_gen_valid, callbacks = callbacks, 
                                        epochs = epochs, validation_steps = steps_per_epoch_val)
