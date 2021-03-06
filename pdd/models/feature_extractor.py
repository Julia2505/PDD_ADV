from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.layers import GlobalAveragePooling2D

import random
import tensorflow as tf


def conv_block(n_filters, 
               filter_size,
               activation='relu',
               #
               #activation=random.choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu']),
               #activation=random.choice(['relu', 'sigmoid', 'tanh', 'elu', 'selu']),
               #
               l1_reg=0, 
               l2_reg=0, 
               dropout=0, 
               batch_norm=False):
  #activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])

    def _conv_block(inputs):
        # don't use bias, if batch_normalization
        bias=True if batch_norm else False

        x = Conv2D(n_filters, filter_size, use_bias=bias,
            kernel_regularizer=L1L2(l1_reg, l2_reg))(inputs)
        x = Activation(activation)(x)
        #x = Activation(random.choice(['relu', 'sigmoid', 'tanh', 'elu', 'selu']))(x)
        
        if batch_norm:
            x = BatchNormalization()(x)

        elif dropout > 0:
            x = Dropout(rate=dropout)(x)

        return MaxPool2D()(x)

    return _conv_block

#@tf.function
def get_pretrained_feature_extractor(img_shape):
    #inputs = Input(input_shape)
    base_model=MobileNetV2(input_shape=img_shape, include_top=False)
    base_model.summary()
    #x=base_model.output
    #x = GlobalAveragePooling2D()(x)
    #x = BatchNormalization()(x)
    #x = Dense(1280, activation='relu')(x)
    #x = BatchNormalization()(x)
    #encoded = Dense(1024, activation='sigmoid')(x)
    #return Model(inputs=base_model.input, outputs=encoded)
  
  #random.randint(512, 1024)
