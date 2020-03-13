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
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import random


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


def get_pretrained_feature_extractor(input_shape):
    inputs = Input(input_shape)
    input_tensor = Input((256, 256))
    #base_model=MobileNet(input_shape=inputs, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=input_tensor, pooling=None, classes=None)
    base_model=MobileNetV2(input_shape=inputs, alpha=1.0, include_top=False, weights=None, input_tensor=input_tensor, pooling=None, classes=None)
    x=base_model.output
    encoded = Dense(1024, activation='sigmoid')(x)
    return Model(inputs, encoded)
  
  #random.randint(512, 1024)
