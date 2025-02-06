#============================================================
#
#  Deep Learning BLW Filtering
#  Deep Learning models
#
#  author: Francisco Perdigon Romero, Wesley Chorney, and Ahmed Shaheen
#  email: ahmed.shaheen@oulu.fi
#  github id: AhmedAShaheen
#
#===========================================================

import numpy as np
from scipy import signal
import math
import random

import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Dropout , GroupNormalization , BatchNormalization, concatenate, Input, Conv2DTranspose, Lambda, LSTM, Layer, MaxPool1D, Conv1DTranspose, Flatten

from tensorflow.keras.layers import Activation, GRU, Reshape, Embedding, GlobalAveragePooling1D, Multiply, Bidirectional
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import LeakyReLU
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(1234)
np.random.seed(1234)
random.seed(1234)


######################################################
from backend import in_train_phase
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/backend.py#L4511

########################################################


########################################################
############## IMPLEMENT DeepFilter ####################
########################################################
def LANLFilter_module(x, layers):
    LB0 = Conv1D(filters=int(layers / 8),
                 kernel_size=3,
                 activation='linear',
                 strides=1,
                 padding='same')(x)
    LB1 = Conv1D(filters=int(layers / 8),
                kernel_size=5,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 8),
                kernel_size=9,
                activation='linear',
                strides=1,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 8),
                kernel_size=15,
                activation='linear',
                strides=1,
                padding='same')(x)

    NLB0 = Conv1D(filters=int(layers / 8),
                  kernel_size=3,
                  activation='relu',
                  strides=1,
                  padding='same')(x)
    NLB1 = Conv1D(filters=int(layers / 8),
                 kernel_size=5,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 8),
                 kernel_size=9,
                 activation='relu',
                 strides=1,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 8),
                 kernel_size=15,
                 activation='relu',
                 strides=1,
                 padding='same')(x)

    x = concatenate([LB0, LB1, LB2, LB3, NLB0, NLB1, NLB2, NLB3])

    return x


def LANLFilter_module_dilated(x, layers):
    LB1 = Conv1D(filters=int(layers / 6),
                kernel_size=5,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB2 = Conv1D(filters=int(layers / 6),
                kernel_size=9,
                activation='linear',
                dilation_rate=3,
                padding='same')(x)
    LB3 = Conv1D(filters=int(layers / 6),
                kernel_size=15,
                dilation_rate=3,
                activation='linear',
                padding='same')(x)

    NLB1 = Conv1D(filters=int(layers / 6),
                 kernel_size=5,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB2 = Conv1D(filters=int(layers / 6),
                 kernel_size=9,
                 activation='relu',
                 dilation_rate=3,
                 padding='same')(x)
    NLB3 = Conv1D(filters=int(layers / 6),
                 kernel_size=15,
                 dilation_rate=3,
                 activation='relu',
                 padding='same')(x)

    x = concatenate([LB1, LB2, LB3, NLB1, NLB2, NLB3])
    # x = BatchNormalization()(x)

    return x

def deep_filter_model_I_LANL_dilated(signal_size=512):
    # TODO: Make the doc

    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    tensor = LANLFilter_module(input, 64)
    tensor = Dropout(0.4, seed=1234)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 64)
    tensor = Dropout(0.4, seed=1234)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 32)
    tensor = Dropout(0.4, seed=1234)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 32)
    tensor = Dropout(0.4, seed=1234)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module(tensor, 16)
    tensor = Dropout(0.4, seed=1234)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LANLFilter_module_dilated(tensor, 16)
    tensor = Dropout(0.4, seed=1234)(tensor)
    tensor = BatchNormalization()(tensor)
    predictions = Conv1D(filters=1,
                    kernel_size=9,
                    activation='linear',
                    strides=1,
                    padding='same')(tensor)

    model = Model(inputs=[input], outputs=predictions)

    return model

########################################################
############## IMPLEMENT CNN DAEs ######################
########################################################
def Conv1DTranspose2(input_tensor, filters, kernel_size, strides=2, activation='relu', padding='same'):
    """
        https://stackoverflow.com/a/45788699

        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = Conv1DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation=activation,
                        strides=strides,
                        padding=padding)(input_tensor)
    return x

def FCN_DAE(signal_size=512):
    # Implementation of FCN_DAE approach presented in
    # Chiang, H. T., Hsieh, Y. Y., Fu, S. W., Hung, K. H., Tsao, Y., & Chien, S. Y. (2019).
    # Noise reduction in ECG signals using fully convolutional denoising autoencoders.
    # IEEE Access, 7, 60806-60813.

    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    x = Conv1D(filters=40,
               input_shape=(512, 1),
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(input)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=40,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=1,
               kernel_size=16,
               activation='elu',
               strides=1,
               padding='same')(x)

    x = BatchNormalization()(x)

    # Keras has no 1D Traspose Convolution, instead we use Conv2DTranspose function
    # in a souch way taht is mathematically equivalent
    
    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=40,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=40,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    predictions = Conv1DTranspose2(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='linear',
                        strides=1,
                        padding='same')

    model = Model(inputs=[input], outputs=predictions)
    return model

def CNN_DAE(signal_size=512):
    
    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    x = Conv1D(filters=40,
               input_shape=(512, 1),
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(input)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=20,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=40,
               kernel_size=16,
               activation='elu',
               strides=2,
               padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(filters=1,
               kernel_size=16,
               activation='elu',
               strides=1,
               padding='same')(x)

    x = BatchNormalization()(x)

    # Keras has no 1D Traspose Convolution, instead we use Conv2DTranspose function
    # in a souch way taht is mathematically equivalent
    x = Conv1DTranspose2(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='elu',
                        strides=1,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=40,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=20,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')

    x = BatchNormalization()(x)

    x = Conv1DTranspose2(input_tensor=x,
                        filters=1,
                        kernel_size=16,
                        activation='elu',
                        strides=2,
                        padding='same')
    x = Flatten()(x)
    x = BatchNormalization()(x)

    x = Dense(signal_size // 2,
              activation='elu')(x)

    x = BatchNormalization()(x)
    x = Dropout(rate=0.5, seed=1234)(x)
    
    predictions = Dense(signal_size, activation='linear')(x)
    predictions = Lambda(lambda x: K.expand_dims(x, axis=2))(predictions) #correct dimnesions
    model = Model(inputs=[input], outputs=predictions)
    return model


########################################################
############## IMPLEMENT DRNN ##########################
########################################################

def DRNN_denoising(signal_size=512):
    # Implementation of DRNN approach presented in
    # Antczak, K. (2018). Deep recurrent neural networks for ECG signal denoising.
    # arXiv preprint arXiv:1807.11551.

    model = Sequential()
    model.add(LSTM(64, input_shape=(signal_size, 1), return_sequences=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    return model


########################################################
########### IMPLEMENT Vanilla AE #######################
########################################################

def VanillaAutoencoder(signal_size=512):
    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)
    
    x = Flatten()(input)
    x = Dense(signal_size // 2)(x)
    x = LeakyReLU()(x)
    x = Dense(signal_size // 4)(x)
    x = LeakyReLU()(x)
    x = Dense(signal_size // 8)(x)
    x = LeakyReLU()(x)
    x = Dense(signal_size // 16)(x)
    x = LeakyReLU()(x)
    x = Dense(signal_size // 8)(x)
    x = LeakyReLU()(x)
    x = Dense(signal_size // 4)(x)
    x = LeakyReLU()(x)
    x = Dense(signal_size // 2)(x)
    x = LeakyReLU()(x)
    x = Dense(signal_size)(x)
    x = LeakyReLU()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(x) #correct dimnesions
    model = Model(inputs=[input], outputs=x)
    return model
        
########################################################
######## IMPLEMENT ECA attention module ################
########################################################

class SpatialGate(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(SpatialGate, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation 
        self.transpose = transpose
        
    def build(self,input_shape):#=(None, signal_size=512, 1)
        # tf.print('calling build method spatial')
        self.conv = Conv1D(filters = self.filters, kernel_size = self.kernel_size, padding='same',
                           # input_shape=input_shape, #int(input_shape[-1]),
                           activation=self.activation)
    def call(self, x):
        #if transpose, switch the data to (batch, steps, channels)
        # tf.print('calling call method spatial')
        if self.transpose:
            x = tf.transpose(x, [0, 2, 1])
        avg_ = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_ = tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.concat([avg_, max_], axis=-1)
        out = self.conv(x)
        if self.transpose:
            out = tf.transpose(out, [0, 2, 1])
        return out


class ChannelGate(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(ChannelGate, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.transpose = transpose
        
    def build(self,input_shape):#https://www.tensorflow.org/api_docs/python/tf/keras/Layer(=(None, signal_size=512, 1))
        # tf.print('calling build method channel')
        self.conv = Conv1D(filters = self.filters, kernel_size = self.kernel_size,
                           # input_shape=input_shape, #int(input_shape[-1]),
                           activation=self.activation,
                           padding='same')
    def call(self, x):
        #if transpose, switch the data to (batch, steps, channels)
        if self.transpose:
            x = tf.transpose(x, [0, 2, 1])
        x = tf.reduce_mean(x, axis=1, keepdims=True)
        x = tf.transpose(x, [0, 2, 1])
        out = self.conv(x)
        out = tf.transpose(out, [0, 2, 1])
        if self.transpose:
            out = tf.transpose(out, [0, 2, 1])
        return out
        

class CBAM(tf.keras.layers.Layer):##ks.layers.Layer
    def __init__(self, c_filters, c_kernel, c_input, c_transpose,
                 s_filters, s_kernel, s_input, s_transpose, spatial=True):
        super(CBAM, self).__init__()
        self.spatial = spatial
        self.c_filters = c_filters
        self.c_kernel = c_kernel 
        self.c_input = c_input
        self.c_transpose = c_transpose
        self.s_filters = s_filters
        self.s_kernel = s_kernel 
        self.s_input = s_input
        self.s_transpose = s_transpose
        
    def build(self, input_shape): # = =(None, signal_size=512, 1)
        self.channel_attention = ChannelGate(self.c_filters, self.c_kernel, input_shape=self.c_input, transpose=self.c_transpose)
        self.spatial_attention = SpatialGate(self.s_filters, self.s_kernel, input_shape=self.s_input, transpose=self.s_transpose)

    def call(self, x):
        channel_mask = self.channel_attention(x)
        x = channel_mask * x
        if self.spatial:
            spatial_mask = self.spatial_attention(x)
            x = spatial_mask * x
        return x


########################################################
################ IMPLEMENT ACDAE #######################
########################################################

class EncoderBlock(Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation='LeakyReLU'):
        super(EncoderBlock, self).__init__()
        if input_size is not None:
            self.conv = Conv1D(
                channels,
                kernel_size,
                input_shape=input_size,
                padding='same',
                #activation=activation
            )
        else:
            self.conv = Conv1D(
                channels,
                kernel_size,
                padding='same',
                #activation=activation
            )
        self.maxpool = MaxPool1D(
            padding='same',
            strides=2
        )
        self.activation = LeakyReLU()
    def call(self, x):
        output = self.conv(x)
        output = self.activation(output)
        output = self.maxpool(output)
        return output

class AttentionDeconvECA(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation='LeakyReLU',
                 strides=2, padding='same'):
        super(AttentionDeconvECA, self).__init__()
        self.deconv = Conv1DTranspose(
            channels,
            kernel_size,
            strides=strides,
            padding=padding,
            #activation=activation
        )
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            7,
            (signal_size, 1),
            False,
            spatial=False
        )
        self.activation = LeakyReLU()
        
    def call(self, x):
        x = self.deconv(x)
        x = self.activation(x)
        output = self.attention(x)
        return output


# class ECASkipDAE(tf.keras.Model):
#     def __init__(self, signal_size=512):
#         super(ECASkipDAE, self).__init__()
#         self.b1 = EncoderBlock(signal_size, 16, input_size=(signal_size, 1), kernel_size=13)
#         self.b2 = EncoderBlock(signal_size//2, 32, kernel_size=7)
#         self.b3 = EncoderBlock(signal_size//4, 64, kernel_size=7)
#         self.b4 = EncoderBlock(signal_size//8, 64, kernel_size=7)
#         self.b5 = EncoderBlock(signal_size//16, 1, kernel_size=7) #32
#         self.d5 = AttentionDeconvECA(signal_size//16, 64, kernel_size=7)
#         self.d4 = AttentionDeconvECA(signal_size//8, 64, kernel_size=7)
#         self.d3 = AttentionDeconvECA(signal_size//4, 32, kernel_size=7)
#         self.d2 = AttentionDeconvECA(signal_size//2, 16, kernel_size=7)
#         self.d1 = AttentionDeconvECA(signal_size, 1, activation='linear', kernel_size=13)
#         self.dense = ks.layers.Dense(signal_size)

#     def encode(self, x):
#         encoded = self.b1(x)
#         encoded = self.b2(encoded)
#         encoded = self.b3(encoded)
#         encoded = self.b4(encoded)
#         encoded = self.b5(encoded)
#         return encoded

#     def decode(self, x):
#         decoded = self.d5(x)
#         decoded = self.d4(decoded)
#         decoded = self.d3(decoded)
#         decoded = self.d2(decoded)
#         decoded = self.d1(decoded)
#         return decoded

#     def call(self, x):
#         enc1 = self.b1(x)
#         enc2 = self.b2(enc1)
#         enc3 = self.b3(enc2)
#         enc4 = self.b4(enc3)
#         enc5 = self.b5(enc4)
#         dec5 = self.d5(enc5)
#         dec4 = self.d4(dec5 + enc4)
#         dec3 = self.d3(dec4 + enc3)
#         dec2 = self.d2(dec3 + enc2)
#         dec1 = self.d1(dec2 + enc1)
#         return dec1

def ECASkipDAE(signal_size=512):
    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)
    
    enc1 = EncoderBlock(signal_size, 16, input_size=(signal_size, 1), kernel_size=13)(input)
    enc2 = EncoderBlock(signal_size//2, 32, kernel_size=7)(enc1)
    enc3 = EncoderBlock(signal_size//4, 64, kernel_size=7)(enc2)
    enc4 = EncoderBlock(signal_size//8, 64, kernel_size=7)(enc3)
    enc5 = EncoderBlock(signal_size//16, 1, kernel_size=7)(enc4) #32
    dec5 = AttentionDeconvECA(signal_size//16, 64, kernel_size=7)(enc5)
    dec4 = AttentionDeconvECA(signal_size//8, 64, kernel_size=7)(dec5 + enc4)
    dec3 = AttentionDeconvECA(signal_size//4, 32, kernel_size=7)(dec4 + enc3)
    dec2 = AttentionDeconvECA(signal_size//2, 16, kernel_size=7)(dec3 + enc2)
    dec1 = AttentionDeconvECA(signal_size, 1, activation='linear', kernel_size=13)(dec2 + enc1)
    # self.dense = ks.layers.Dense(signal_size)

    model = Model(inputs=[input], outputs=dec1)
    return model

########################################################
############### IMPLEMENT CDAE-BAM #####################
########################################################

class AttentionBlockBN(Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None):
        super(AttentionBlockBN, self).__init__()
        if input_size is not None:
            self.conv = Conv1D(
                channels,
                kernel_size,
                input_shape=input_size,
                padding='same',
                activation=None
            )
        else:
            self.conv = Conv1D(
                channels,
                kernel_size,
                padding='same',
                activation=None
            )
        self.activation = tf.keras.layers.LeakyReLU()
        self.bn = BatchNormalization()
        self.dp = Dropout(rate=0.001, seed=1234) #rate=0.1 for qtdb  rate=0.001 for other datasets
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            7,
            (signal_size, 1),
            False
        )
        self.maxpool = MaxPool1D(
            padding='same',
            strides=2
        )

    def call(self, x):
        output = self.conv(x)
        output = self.activation(self.bn(output))
        output = self.dp(output)
        output = self.attention(output)
        output = self.maxpool(output)
        return output

class AttentionDeconvBN(tf.keras.layers.Layer):
    def __init__(self, signal_size, channels, kernel_size=16,
                 input_size=None, activation='LeakyReLU',
                 strides=2, padding='same'):
        super(AttentionDeconvBN, self).__init__()
        self.deconv = Conv1DTranspose(
            channels,
            kernel_size,
            strides=strides,
            padding=padding,
        )
        self.bn = BatchNormalization()
        if activation == 'LeakyReLU':
            self.activation = tf.keras.layers.LeakyReLU()
        else:
            self.activation = None
        self.dp = Dropout(rate=0.001, seed=1234) #rate=0.1 for qtdb rate=0.001 for other datasets
        self.attention = CBAM(
            1,
            3,
            (channels, 1),
            False,
            1,
            7,
            (signal_size, 1),
            False
        )

    def call(self, x):
        output = self.deconv(x)
        output = self.bn(output)
        if self.activation is not None:
            output = self.activation(output)
        output = self.dp(output)
        output = self.attention(output)
        return output

# class AttentionSkipDAE2(tf.keras.Model):
#     def __init__(self, signal_size=512):
#         super(AttentionSkipDAE2, self).__init__()
#         self.b1 = AttentionBlockBN(signal_size, 16, input_size=(signal_size, 1))
#         self.b2 = AttentionBlockBN(signal_size//2, 32)
#         self.b3 = AttentionBlockBN(signal_size//4, 64)
#         self.b4 = AttentionBlockBN(signal_size//8, 64)
#         self.b5 = AttentionBlockBN(signal_size//16, 1) #32
#         self.d5 = AttentionDeconvBN(signal_size//16, 64)
#         self.d4 = AttentionDeconvBN(signal_size//8, 64)
#         self.d3 = AttentionDeconvBN(signal_size//4, 32)
#         self.d2 = AttentionDeconvBN(signal_size//2, 16)
#         self.d1 = AttentionDeconvBN(signal_size, 1, activation='linear')

#     def encode(self, x):
#         encoded = self.b1(x)
#         encoded = self.b2(encoded)
#         encoded = self.b3(encoded)
#         encoded = self.b4(encoded)
#         encoded = self.b5(encoded)
#         return encoded

#     def decode(self, x):
#         decoded = self.d5(x)
#         decoded = self.d4(decoded)
#         decoded = self.d3(decoded)
#         decoded = self.d2(decoded)
#         decoded = self.d1(decoded)
#         return decoded

#     def call(self, x):
#         enc1 = self.b1(x)
#         enc2 = self.b2(enc1)
#         enc3 = self.b3(enc2)
#         enc4 = self.b4(enc3)
#         enc5 = self.b5(enc4)
#         dec5 = self.d5(enc5)
#         dec4 = self.d4(dec5 + enc4)
#         dec3 = self.d3(dec4 + enc3)
#         dec2 = self.d2(dec3 + enc2)
#         dec1 = self.d1(dec2 + enc1)
#         return dec1
        
def AttentionSkipDAE2(signal_size=512):
        input_shape = (signal_size, 1)
        input = Input(shape=input_shape)

        enc1 = AttentionBlockBN(signal_size, 16, input_size=(signal_size, 1))(input)
        enc2 = AttentionBlockBN(signal_size//2, 32)(enc1)
        enc3 = AttentionBlockBN(signal_size//4, 64)(enc2)
        enc4 = AttentionBlockBN(signal_size//8, 64)(enc3)
        enc5 = AttentionBlockBN(signal_size//16, 1)(enc4) #32
        dec5 = AttentionDeconvBN(signal_size//16, 64)(enc5)
        dec4 = AttentionDeconvBN(signal_size//8, 64)(dec5 + enc4)
        dec3 = AttentionDeconvBN(signal_size//4, 32)(dec4 + enc3)
        dec2 = AttentionDeconvBN(signal_size//2, 16)(dec3 + enc2)
        dec1 = AttentionDeconvBN(signal_size, 1, activation='linear')(dec2 + enc1)

        model = Model(inputs=[input], outputs=dec1)
        return model
    
########################################################
################ IMPLEMENT TCDAE #######################
########################################################

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
    emb = tf.reshape(emb, (*emb.shape[:-2], -1))
    return emb


class TFPositionalEncoding1D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.
        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".
        """
        super(TFPositionalEncoding1D, self).__init__()

        self.channels = int(np.ceil(channels / 2) * 2)
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )
        self.cached_penc = None

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(inputs.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc

        self.cached_penc = None
        _, x, org_channels = inputs.shape

        dtype = self.inv_freq.dtype
        pos_x = tf.range(x, dtype=dtype)
        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        emb = tf.expand_dims(get_emb(sin_inp_x), 0)
        emb = emb[0]  # A bit of a hack
        self.cached_penc = tf.repeat(
            emb[None, :, :org_channels], tf.shape(inputs)[0], axis=0
        )

        return self.cached_penc
        
        
def transformer_encoder(inputs,head_size,num_heads,ff_dim,dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x= layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout, seed=1234)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)  #sigmoid: Previously used, gelu: You can try it
    x = layers.Dropout(dropout, seed=1234)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def spatial_attention(inputs):
    attention = tf.keras.layers.Dense(1, activation='tanh')(inputs)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.Reshape((-1, 1))(attention)
    return attention
    
def attention_module(inputs, filters):
    x = tf.keras.layers.Conv1D(filters, kernel_size=1, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
    attention = tf.keras.layers.GlobalAveragePooling1D()(x)
    attention = tf.keras.layers.Dense(filters, activation='sigmoid')(attention)
    attention = tf.keras.layers.Reshape((1, filters))(attention)
    scaled_inputs = tf.keras.layers.Multiply()([inputs, attention])
    return scaled_inputs


class AddGatedNoise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddGatedNoise, self).__init__(**kwargs)

    def call(self, x, training=None):
        # During training, random noise is used
        noise = tf.random.uniform(shape=tf.shape(x), minval=-1, maxval=1)
        return in_train_phase(x * (1 + noise), x, training=training)        


def Transformer_DAE(signal_size = 512,head_size=64,num_heads=8,ff_dim=64,num_transformer_blocks=6, dropout=0):   ###paper 1 model

    ks = 13   #orig 13

    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)

    x0 = Conv1D(filters=16,
                input_shape=(input_shape, 1),
                kernel_size=ks,
                activation='linear',  # Using linear activation function
                strides=2,
                padding='same')(input)

    # Use a custom layer to add multiplicative noise, only during training
    x0 = AddGatedNoise()(x0)

    # Apply sigmoid activation function
    x0 = layers.Activation('sigmoid')(x0)
    # x0 = Dropout(0.3)(x0)
    x0_ = Conv1D(filters=16,
               input_shape=(input_shape, 1),
               kernel_size=ks,
               activation=None,
               strides=2,
               padding='same')(input)
    # x0_ = Dropout(0.3)(x0_)
    xmul0 = Multiply()([x0,x0_])

    xmul0 = BatchNormalization()(xmul0)

    x1 = Conv1D(filters=32,
                kernel_size=ks,
                activation='linear',  # Using linear activation function
                strides=2,
                padding='same')(xmul0)

    # Use a custom layer to add multiplicative noise, only during training
    x1 = AddGatedNoise()(x1)

    # Apply sigmoid activation function
    x1 = layers.Activation('sigmoid')(x1)

    # x1 = Dropout(0.3)(x1)
    x1_ = Conv1D(filters=32,
               kernel_size=ks,
               activation=None,
               strides=2,
               padding='same')(xmul0)
    # x1_ = Dropout(0.3)(x1_)
    xmul1 = Multiply()([x1, x1_])
    xmul1 = BatchNormalization()(xmul1)

    x2 = Conv1D(filters=64,
               kernel_size=ks,
               activation='linear',
               strides=2,
               padding='same')(xmul1)
    x2 = AddGatedNoise()(x2)
    
    # Apply sigmoid activation function
    x2 = layers.Activation('sigmoid')(x2)
    
    x2_ = Conv1D(filters=64,
               kernel_size=ks,
               activation='elu',
               strides=2,
               padding='same')(xmul1)
    
    xmul2 = Multiply()([x2, x2_])

    xmul2 = BatchNormalization()(xmul2)

    #Positional encoding
    position_embed = TFPositionalEncoding1D(signal_size)
    x3 = xmul2+position_embed(xmul2)
    
    for _ in range(num_transformer_blocks):
        x3 = transformer_encoder(x3,head_size,num_heads,ff_dim, dropout)
    
    x4 = x3
    
    x5 = Conv1DTranspose(
                        filters=64,
                        kernel_size=ks,
                        activation='elu',
                        strides=1,
                        padding='same')(x4)
    x5 = x5+xmul2
    x5 = BatchNormalization()(x5)

    x6 = Conv1DTranspose(
                        filters=32,
                        kernel_size=ks,
                        activation='elu',
                        strides=2,
                        padding='same')(x5)
    x6 = x6+xmul1
    x6 = BatchNormalization()(x6)

    x7 = Conv1DTranspose(
                        filters=16,
                        kernel_size=ks,
                        activation='elu',
                        strides=2,
                        padding='same')(x6)

    x7 = x7 + xmul0 #res

    x8 = BatchNormalization()(x7)
    predictions = Conv1DTranspose(
                        # input_tensor=x8,
                        filters=1,
                        kernel_size=ks,
                        activation='linear',
                        strides=2,
                        padding='same')(x8)

    model = Model(inputs=[input], outputs=predictions)
    return model
    
########################################################
############## IMPLEMENT FGDAE ######################
########################################################

class SelfONNLayer1D(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, stride=1, padding='same', 
                 dilation=1, use_bias=True, q=1):
        super(SelfONNLayer1D, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding.upper()  # Ensure uppercase for TensorFlow compatibility
        self.dilation = dilation
        self.q = q
        self.use_bias = use_bias
        self.in_channels = None  # We'll infer this in the `build` method

    def build(self, input_shape):
        # Infer in_channels from input_shape
        self.in_channels = input_shape[-1]  # Last dimension is the number of channels
        
        # Initialize weights based on in_channels
        self.weights_onn = self.add_weight(shape=(self.q, self.out_channels, self.in_channels, self.kernel_size),
                                           initializer='glorot_uniform', trainable=True, name='weights_onn')
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.out_channels,), initializer='zeros', trainable=True, name='bias')
        else:
            self.bias = None
            
        
    def call(self, x):
        # Apply nonlinearity to x
        x = tf.concat([tf.pow(x, i) for i in range(1, self.q + 1)], axis=-1)

        # Reshape weights for 1D convolution
        w = tf.reshape(self.weights_onn, (self.q * self.in_channels, self.out_channels, self.kernel_size))
        w = tf.transpose(w, perm=[2, 0, 1])  # TensorFlow expects [kernel_size, in_channels, out_channels]

        # Perform 1D convolution
        x = tf.nn.conv1d(x, w, stride=self.stride, padding=self.padding, dilations=self.dilation)

        # Add bias if needed
        if self.bias is not None:
            x = tf.nn.bias_add(x, self.bias)
        
        return x

    def reset_parameters(self):
        # Initialize weights with Xavier uniform (equivalent to PyTorch's xavier_uniform_)
        initializer = tf.keras.initializers.GlorotUniform()
        for q in range(self.q):
            # Reset weights per q value
            self.weights_onn[q].assign(initializer(self.weights_onn[q].shape))  
        # Reset bias if needed
        if self.bias is not None:
            bound = 0.01
            self.bias.assign(tf.random.uniform(self.bias.shape, -bound, bound))


class GatedConv(Layer):
    def __init__(self, filters, kernel_size, input_shape=None, strides=1):
        super(GatedConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        if input_shape is not None:
            self.conv_gate = Conv1D(filters = self.filters, input_shape=(input_shape, 1), kernel_size = self.kernel_size, strides=self.strides, padding='same', activation='sigmoid')
            self.conv_input = Conv1D(filters = self.filters, input_shape=(input_shape, 1), kernel_size = self.kernel_size, strides=self.strides, padding='same', activation=None)
        else:
            self.conv_gate = Conv1D(filters = self.filters, kernel_size = self.kernel_size, strides=self.strides, padding='same', activation='sigmoid')
            self.conv_input = Conv1D(filters = self.filters, kernel_size = self.kernel_size, strides=self.strides, padding='same', activation=None)
        self.dp = Dropout(rate=0.001, seed=1234)
        
    def call(self, x):
        x_g = self.conv_gate(x)
        if x_g.shape[2]>1:
            x_g = self.dp(x_g)
        x_i = self.conv_input(x)
        x_out = Multiply()([x_g,x_i])
        return x_out 


class GatedSelfONN(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, q=1):
        super(GatedSelfONN, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.q = q  # Degree of nonlinearity in SelfONN
        self.conv_gate = SelfONNLayer1D(out_channels=self.filters, 
                                        kernel_size=self.kernel_size, stride=self.strides, 
                                        padding='same', q=self.q)
        self.conv_input = SelfONNLayer1D(out_channels=self.filters, 
                                         kernel_size=self.kernel_size, stride=self.strides, 
                                         padding='same', q=self.q)
        self.dp = Dropout(rate=0.001*q, seed=1234)
        
    def call(self, x):
        # Gate processing using sigmoid activation
        x_g = self.conv_gate(x)
        x_g = tf.keras.activations.sigmoid(x_g)
        if x_g.shape[2]>1:
            x_g = self.dp(x_g)
        # Input processing with SelfONN, no activation
        x_i = self.conv_input(x)
        # Multiply the gated and input outputs
        x_out = Multiply()([x_g, x_i])
        return x_out


class GatedDeConv(Layer):
    def __init__(self, filters, kernel_size, input_shape=None, strides=1):
        super(GatedDeConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        if input_shape is not None:
            self.deconv_gate = Conv1DTranspose(self.filters, input_shape=(input_shape, 1), kernel_size = self.kernel_size, strides=self.strides, padding='same', activation='sigmoid')
            self.deconv_input = Conv1DTranspose(self.filters, input_shape=(input_shape, 1), kernel_size = self.kernel_size, strides=self.strides, padding='same', activation=None)
        else:
            self.deconv_gate = Conv1DTranspose(self.filters, kernel_size = self.kernel_size, strides=self.strides, padding='same', activation='sigmoid')
            self.deconv_input = Conv1DTranspose(self.filters, kernel_size = self.kernel_size, strides=self.strides, padding='same', activation=None)
        self.dp = Dropout(rate=0.001, seed=1234)
        
    def call(self, x):
        x_g = self.deconv_gate(x)
        if x_g.shape[2]>1:
            x_g = self.dp(x_g)
        x_i = self.deconv_input(x)
        x_out = Multiply()([x_g,x_i])
        return x_out 


class fullSpatialGate(Layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(fullSpatialGate, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation 
        self.transpose = transpose
        
    def build(self,input_shape):#=(None, signal_size=512, 1)
        self.conv = Conv1D(filters = self.filters, 
                           kernel_size = self.kernel_size, 
                           padding='same',
                           activation=self.activation)
    def call(self, x):
        #if transpose, switch the data to (batch, steps, channels)
        if self.transpose:
            x = tf.transpose(x, [0, 2, 1])
        avg_ = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_ = tf.reduce_max(x, axis=-1, keepdims=True)
        x = tf.concat([avg_, max_], axis=-1)
        out = self.conv(x)
        if self.transpose:
            out = tf.transpose(out, [0, 2, 1])
        return out


class fullChannelGate(Layer):
    def __init__(self, filters, kernel_size, input_shape=None, activation='sigmoid', transpose=False):
        super(fullChannelGate, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.transpose = transpose
        
    def build(self,input_shape):#https://www.tensorflow.org/api_docs/python/tf/keras/Layer(=(None, signal_size=512, 1))
        self.conv = Conv1D(filters = self.filters, 
                           kernel_size = self.kernel_size,
                           activation=self.activation,
                           padding='same')
        self.dp = Dropout(rate=0.001, seed=1234)
        
    def call(self, x):
        #if transpose, switch the data to (batch, steps, channels)
        if self.transpose:
            x = tf.transpose(x, [0, 2, 1])
        avg_ = tf.reduce_mean(x, axis=1, keepdims=True)
        max_ = tf.reduce_max(x, axis=1, keepdims=True)
        avg_ = tf.transpose(avg_, [0, 2, 1])
        max_ = tf.transpose(max_, [0, 2, 1])
        out_avg = self.conv(avg_)
        out_max = self.conv(max_)
        out = tf.transpose(out_avg+out_max, [0, 2, 1])
        if out.shape[2]>1:
            out = self.dp(out)
        if self.transpose:
            out = tf.transpose(out, [0, 2, 1])
        return out
        

class fullCBAM(Layer):##ks.layers.Layer
    def __init__(self, c_filters, c_kernel, c_input, c_transpose, s_filters, s_kernel, s_input, s_transpose, spatial=True):
        super(fullCBAM, self).__init__()
        self.c_filters = c_filters
        self.c_kernel = c_kernel 
        self.c_input = c_input
        self.c_transpose = c_transpose
        self.spatial = spatial
        if self.spatial:
            self.s_filters = s_filters
            self.s_kernel = s_kernel 
            self.s_input = s_input
            self.s_transpose = s_transpose
        
    def build(self, input_shape): # = =(None, signal_size=512, 1)
        self.channel_attention = fullChannelGate(self.c_filters, self.c_kernel, input_shape=self.c_input, transpose=self.c_transpose)
        if self.spatial:
            self.spatial_attention = fullSpatialGate(self.s_filters, self.s_kernel, input_shape=self.s_input, transpose=self.s_transpose)

    def call(self, x):
        channel_mask = self.channel_attention(x)
        x = channel_mask * x
        if self.spatial:
            spatial_mask = self.spatial_attention(x)
            x = spatial_mask * x
        return x


# Adaptive Attention Gate
class AdaptiveAttentionGate(Layer):
    def __init__(self, input_shape, filters, kernel_size=1, strides=1):
        super(AdaptiveAttentionGate, self).__init__() 
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
    
    def build(self, input_shape): # = =(None, signal_size=512, 1)
        self.conv_gate = Conv1D(filters = self.filters,
                                input_shape=(self.input_shape, 1), 
                                kernel_size = self.kernel_size, 
                                strides=self.strides,
                                padding='same',
                                activation=None)
                                
        self.conv_input = Conv1D(filters = self.filters,
                                 input_shape=(self.input_shape, 1), 
                                 kernel_size = self.kernel_size, 
                                 strides=self.strides,
                                 padding='same',
                                 activation=None)
        self.dp = Dropout(rate=0.001, seed=1234)
        
    def call(self, x, g):
        # Apply linear transformations to the gating signal (g) and the feature map (x)
        gating_signal = self.conv_gate(g)
        gating_signal = self.dp(gating_signal)
        input_signal = self.conv_input(x)
        # Sum the transformed feature maps, followed by sigmoid
        attention_weights = tf.keras.activations.sigmoid(input_signal + gating_signal)
        # Element-wise multiplication of attention map with the input feature map (x)
        return x * attention_weights

        
class gatedEncoderBlock(Layer):
    def __init__(self, signal_size, channels, kernel_size=9, input_size=None, strides=1, padding='same', activation='LeakyReLU'):
        super(gatedEncoderBlock, self).__init__()
        if input_size is not None:
            self.conv = GatedConv(channels, kernel_size, input_shape=input_size, strides=strides)
        else:
            self.conv = GatedConv(channels, kernel_size, strides=strides)
            
        if activation == 'LeakyReLU':
            self.activation = tf.keras.layers.LeakyReLU()
        else:
            self.activation = None
        self.norm = GroupNormalization(groups=-1)
        self.maxpool = MaxPool1D(padding='same', strides=2)
        self.attention = fullCBAM(1, 3, (channels, 1), False, 1, 7, (signal_size, 1), False, spatial=False)
        
    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.maxpool(x)
        x = self.attention(x)
        return x


class gatedONNEncoderBlock(Layer):
    def __init__(self, signal_size, channels, kernel_size=9, input_size=None, strides=1, padding='same', activation='LeakyReLU', q=2):
        super(gatedONNEncoderBlock, self).__init__()
        
        self.conv = GatedSelfONN(channels, kernel_size, strides=strides, q=q)
        if activation == 'LeakyReLU':
            self.activation = tf.keras.layers.LeakyReLU()
        else:
            self.activation = None
        self.norm = GroupNormalization(groups=-1)
        self.maxpool = MaxPool1D(padding='same', strides=2)
        self.attention = fullCBAM(1, 3, (channels, 1), False, 1, 7, (signal_size, 1), False, spatial=False)
        
    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.maxpool(x)
        x = self.attention(x)
        return x


class gatedDecoderBlock(Layer):
    def __init__(self, signal_size, channels, kernel_size=9, strides=2, activation='LeakyReLU'):
        super(gatedDecoderBlock, self).__init__()
        self.deconv = GatedDeConv(channels, kernel_size, strides=strides)
        if activation == 'LeakyReLU':
            self.activation = tf.keras.layers.LeakyReLU()
        else:
            self.activation = None
        self.norm = GroupNormalization(groups=-1)
        self.attention = fullCBAM(1, 3, (channels, 1), False, 1, 7, (signal_size, 1), False, spatial=False)
        
    def call(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.attention(x)
        return x
        
    
def GatedONNDAE(signal_size=512, q=2):
    input_shape = (signal_size, 1)
    input = Input(shape=input_shape)
    
    # Encoder
    enc1 = gatedONNEncoderBlock(signal_size//2,  16, kernel_size=9, q=q)(input)
    enc2 = gatedONNEncoderBlock(signal_size//4,  32, kernel_size=9, q=q)(enc1)
    enc3 = gatedONNEncoderBlock(signal_size//8,  64, kernel_size=9, q=q)(enc2)
    enc4 = gatedONNEncoderBlock(signal_size//16, 64, kernel_size=9, q=q)(enc3)
    enc5 = gatedONNEncoderBlock(signal_size//32,  1, kernel_size=9, q=q)(enc4) #32
    
    # Decoder
    dec5 = gatedDecoderBlock(signal_size//16,     64, kernel_size=9)(enc5)
    dec5 = AdaptiveAttentionGate(signal_size//16, 64, kernel_size=1, strides=1)(dec5, enc4)
    dec4 = gatedDecoderBlock(signal_size//8,      64, kernel_size=9)(dec5)
    dec4 = AdaptiveAttentionGate(signal_size//8,  64, kernel_size=1, strides=1)(dec4, enc3)
    dec3 = gatedDecoderBlock(signal_size//4,      32, kernel_size=9)(dec4)
    dec3 = AdaptiveAttentionGate(signal_size//4,  32, kernel_size=1, strides=1)(dec3, enc2)
    dec2 = gatedDecoderBlock(signal_size//2,      16, kernel_size=9)(dec3)
    dec2 = AdaptiveAttentionGate(signal_size//2,  16, kernel_size=1, strides=1)(dec2, enc1)
    dec1 = gatedDecoderBlock(signal_size,          1, kernel_size=9, activation='linear')(dec2)
    
    model = Model(inputs=[input], outputs=dec1)
    return model