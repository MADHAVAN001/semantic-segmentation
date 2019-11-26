from keras.backend import image_data_format
from keras.models import Model
from keras.regularizers import l2

from models.BilinearUpSampling import *


class FCN:

    def identity_block(self, kernel_size, filters, stage, block, weight_decay=0., momentum=0.99):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
        """

        def f(input_tensor):
            filters1, filters2, filters3 = filters
            if image_data_format() == 'channels_last':
                bn_axis = 3
            else:
                bn_axis = 1

            convolution_base_name = 'res' + str(stage) + block + '_branch'
            bn_base_name = 'bn' + str(stage) + block + '_branch'

            x = Conv2D(
                filters1,
                (1, 1),
                use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay)
            )(input_tensor)

            x = BatchNormalization(axis=bn_axis, momentum=momentum)(x)
            x = Activation('relu')(x)

            x = Conv2D(
                filters2,
                kernel_size,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay)
            )(x)

            x = BatchNormalization(
                axis=bn_axis,
                momentum=momentum
            )(x)

            x = Activation('relu')(x)

            x = Conv2D(
                filters3,
                (1, 1),
                use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(weight_decay)
            )(x)

            x = BatchNormalization(axis=bn_axis, momentum=momentum)(x)

            x = add([x, input_tensor])
            x = Activation('relu')(x)
            return x

        return f

    def convolution_block(self, kernel_size, filters, stage, block, strides=(2, 2), weight_decay=0., momentum=0.99,
                          epsilon=0.):
        """ Definition of Convolutional Block in a typical Resnet
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
        # Returns
            Output tensor for the block.
        Note that from stage 3,
        the second conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """

        def f(input_tensor):
            filters1, filters2, filters3 = filters

            if image_data_format() == 'channels_last':
                bn_axis = 3
            else:
                bn_axis = 1

            convolution_base_name = 'res' + str(stage) + block + '_branch'
            bn_base_name = 'bn' + str(stage) + block + '_branch'

            x = Conv2D(filters1, (1, 1), use_bias=False, kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay))(input_tensor)
            x = BatchNormalization(axis=bn_axis, momentum=momentum)(x)
            x = Activation('relu')(x)

            x = Conv2D(filters2, kernel_size, strides=strides, padding='same', use_bias=False,
                       kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

            x = BatchNormalization(axis=bn_axis, momentum=momentum)(x)
            x = Activation('relu')(x)

            x = Conv2D(filters3, (1, 1), use_bias=False, kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay))(x)
            x = BatchNormalization(axis=bn_axis, momentum=momentum)(x)

            shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal',
                              kernel_regularizer=l2(weight_decay))(input_tensor)
            shortcut = BatchNormalization(axis=bn_axis, momentum=momentum)(shortcut)

            x = add([x, shortcut])
            x = Activation('relu')(x)
            return x

        return f

    def resnet50(self, input_shape=None, weight_decay=0., momentum=0.9, batch_shape=None, classes=21):
        if batch_shape:
            img_input = Input(batch_shape=batch_shape)
            image_size = batch_shape[1:3]
        else:
            img_input = Input(shape=input_shape)
            image_size = input_shape[0:2]

        bn_axis = 3

        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(
            img_input)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.convolution_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
        x = self.identity_block(3, [64, 64, 256], stage=2, block='b')(x)
        x = self.identity_block(3, [64, 64, 256], stage=2, block='c')(x)

        x = self.convolution_block(3, [128, 128, 512], stage=3, block='a')(x)
        x = self.identity_block(3, [128, 128, 512], stage=3, block='b')(x)
        x = self.identity_block(3, [128, 128, 512], stage=3, block='c')(x)
        x = self.identity_block(3, [128, 128, 512], stage=3, block='d')(x)

        x = self.convolution_block(3, [256, 256, 1024], stage=4, block='a')(x)
        x = self.identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
        x = self.identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
        x = self.identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
        x = self.identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
        x = self.identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

        x = self.convolution_block(3, [512, 512, 2048], stage=5, block='a')(x)
        x = self.identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
        x = self.identity_block(3, [512, 512, 2048], stage=5, block='c')(x)

        x = Conv2D(classes,
                   (1, 1),
                   kernel_initializer='he_normal',
                   activation='linear',
                   padding='valid',
                   strides=(1, 1),
                   kernel_regularizer=l2(weight_decay))(x)

        x = BilinearUpSampling2D(size=(32, 32))(x)

        model = Model(img_input, x)
        return model
