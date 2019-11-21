import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class uNetModel:
    
    def __init__(self, inputWidth, inputHeight,numChannels = 1, numClasses=13, n_filters=16, kernel_size=3, dropout = 0.05, batchnorm=True):
        # Initialize instance
        input_img = Input((inputWidth, inputHeight, numChannels), name='img')
        self.model = self.get_unet(input_img, n_filters=n_filters, ksize = kernel_size, dropout=dropout, batchnorm=batchnorm, numClasses=numClasses)
    
    def argmax_op(self, input):
        return(tf.keras.backend.cast(tf.keras.backend.argmax(input,-1),'float64'))

    def softargmax(self, x, beta=1e3):
        #x = tf.convert_to_tensor(x)
        x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
        return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)
    
    def conv2d_block(self, input_tensor, n_filters, kernel_size, batchnorm):
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                  padding="same")(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x


    def get_unet(self, input_img, n_filters, ksize, dropout, batchnorm, numClasses):
        # contracting path
        c1 = self.conv2d_block(input_img, n_filters=n_filters*1, kernel_size=ksize, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2)) (c1)
        p1 = Dropout(dropout*0.5)(p1)

        c2 = self.conv2d_block(p1, n_filters=n_filters*2, kernel_size=ksize, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2)) (c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters=n_filters*4, kernel_size=ksize, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2)) (c3)
        p3 = Dropout(dropout)(p3)

        c4 = self.conv2d_block(p3, n_filters=n_filters*8, kernel_size=ksize, batchnorm=batchnorm)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        p4 = Dropout(dropout)(p4)
    
        c5 = self.conv2d_block(p4, n_filters=n_filters*16, kernel_size=ksize, batchnorm=batchnorm)
    
        # expansive path
        u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters=n_filters*8, kernel_size=ksize, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters=n_filters*4, kernel_size=ksize, batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters=n_filters*2, kernel_size=ksize, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters=n_filters*1, kernel_size=ksize, batchnorm=batchnorm)
    
        outputs = Conv2D(numClasses, (1, 1), activation='sigmoid') (c9)
        
        #outputs = Lambda(self.argmax_op)(outputs)
        outputs = Lambda(self.softargmax)(outputs)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model

'''
uNetInst = uNetModel(inputWidth=128, inputHeight=128, numChannels=3, numClasses=13) 
#input_img = Input((im_height, im_width, 1), name='img')
#model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

#model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
uNetInst.model.summary()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))
'''
