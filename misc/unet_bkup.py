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
from tensorflow_model_optimization.sparsity import keras as sparsity

class uNetModel:
    
    def __init__(self, inputWidth, inputHeight,numChannels = 1, numClasses=13, n_filters=16, kernel_size=3, dropout = 0.05, batchnorm=True):
        # Initialize instance
        input_img = Input((inputWidth, inputHeight, numChannels), name='img')
        self.prune_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.2, final_sparsity=0.6,begin_step=3000, end_step=75000, frequency=100)}
        self.model = self.get_unet(input_img, n_filters=n_filters, ksize = kernel_size, dropout=dropout, batchnorm=batchnorm, numClasses=numClasses)
    
    def argmax_op(self, input):
        return(tf.keras.backend.cast(tf.keras.backend.argmax(input,-1),'float64'))

    def softargmax(self, x, beta=1e3):
        #x = tf.convert_to_tensor(x)
        x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
        return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)
    
    def conv2d_block(self, n_filters, kernel_size, batchnorm,sparsify = False):
        # first layer
        prune_prm = self.prune_params
        l = tf.keras.layers
        l1 = l.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                  padding="same")
        if(sparsify):
            l1 = sparsity.prune_low_magnitude(l1, **prune_prm)
        if batchnorm:
            l2 = l.BatchNormalization()
        l3 = l.Activation("relu")
        
        l4 = l.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")
        if(sparsify):
            l4 = sparsity.prune_low_magnitude(l4, **prune_prm)
        if batchnorm:
            l5 = l.BatchNormalization()
        l6 = l.Activation("relu")
        return l1, l2, l3, l4, l5, l6


    def get_unet(self, input_img, n_filters, ksize, dropout, batchnorm, numClasses):
        # contracting path
        l = tf.keras.layers
        kernel_size = ksize
        prune_prm = self.prune_params
        c1 = sparsity.prune_low_magnitude(l.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                 padding="same", input_shape = input_img.shape),**prune_prm)
        b1 = l.BatchNormalization()
        a1 = l.Activation("relu")
        
        c2 = sparsity.prune_low_magnitude(l.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                 padding="same"),**prune_prm)
        b2 = l.BatchNormalization()
        a2 = l.Activation("relu")
        
        m1 = l.MaxPooling2D((2, 2))
        d1 = l.Dropout(dropout*0.5)
        
        c3, b3, a3, c4, b4, a4 = self.conv2d_block(n_filters=n_filters*2, kernel_size=ksize, batchnorm=batchnorm,sparsify=True)
        m2 = l.MaxPooling2D((2, 2))
        d2 = l.Dropout(dropout)
        
        c5, b5, a5, c6, b6, a6 = self.conv2d_block(n_filters=n_filters*4, kernel_size=ksize, batchnorm=batchnorm,sparsify=True)
        m3 = l.MaxPooling2D((2, 2))
        d3 = l.Dropout(dropout)
        
        c7, b7, a7, c8, b8, a8 = self.conv2d_block(n_filters=n_filters*8, kernel_size=ksize, batchnorm=batchnorm,sparsify=True)
        m4 = l.MaxPooling2D(pool_size=(2, 2))
        d4 = l.Dropout(dropout)
    
        c9, b9, a9, c10, b10, a10 = self.conv2d_block(n_filters=n_filters*16, kernel_size=ksize, batchnorm=batchnorm,sparsify=True)
    
        # expansive path
        ct1 = l.Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')
        con1 = l.Concatenate([ct1, c8])
        d5 = l.Dropout(dropout)
        c11, b11, a11, c12, b12, a12 = self.conv2d_block(n_filters=n_filters*8, kernel_size=ksize, batchnorm=batchnorm, sparsify=True)

        ct2 = l.Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')
        con2 = l.Concatenate([ct2, c6])
        d6 = l.Dropout(dropout)
        c13, b13, a13, c14, b14, a14 = self.conv2d_block(n_filters=n_filters*4, kernel_size=ksize, batchnorm=batchnorm,sparsify=True)

        ct3 = l.Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')
        con3 = l.Concatenate([ct3, c4])
        d7 = l.Dropout(dropout)
        c15, b15, a15, c16, b16, a16 = self.conv2d_block(n_filters=n_filters*2, kernel_size=ksize, batchnorm=batchnorm, sparsify=True)

        ct4 = l.Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')
        con4 = l.Concatenate(axis=3) #([ct4,c2])

        #con4 = l.Concatenate([ct4, c2], axis=3)
        #con4 = l.Concatenate(axis=3)
        
        d8 = l.Dropout(dropout)
        c17, b17, a17, c18, b18, a18 = self.conv2d_block(n_filters=n_filters*1, kernel_size=ksize, batchnorm=batchnorm,sparsify=True)
    
        c19 = sparsity.prune_low_magnitude(l.Conv2D(numClasses, (1, 1), activation='sigmoid'),**prune_prm)
        
        model = tf.keras.Sequential([c1, b1, a1, c2, b2, a2, m1, d1, 
        c3, b3, a3, c4, b3, a4, m3, d2,
        c5, b5, a5, c6, b6, a6, m3, d3,
        c7, b7, a7, c8, b8, a8, m4, d4,
        c9, b9, a9, c10, b10, a10, 
        ct1, con1, d5,
        c11, b11, a11, c12, b12, a12, 
        ct2, con2, d6,
        c13, b13, a13, c14, b14, a14,
        ct3, con3, d7,
        c15, b15, a15, c16, b16, a16,
        ct4, con4, d8,
        c17, b17, a17, c18, b18, a18,
        c19])
        return model

uNetInst = uNetModel(inputWidth=128, inputHeight=128, numChannels=3, numClasses=13) 
#input_img = Input((im_height, im_width, 1), name='img')
#model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

#model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
uNetInst.model.summary()

'''
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=callbacks,
                    validation_data=(X_valid, y_valid))
'''
