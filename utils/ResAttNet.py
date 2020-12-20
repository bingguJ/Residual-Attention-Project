import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, AveragePooling2D
from tensorflow.keras import Model
from utils.layer_units import *

class ResAttNet():
# this class will help build an residual attention network with arbitrary
# number of attention modules and can specify the residual units parameters
# I tried to make it a subclass of Model, but it didn't work well with my other
# implementation so I resolve to build manually with a build function inside
    def __init__(self,output_size=10, dense_layer_units = 256):
        '''
        Structure inspired by HW2 my_Lenet.py
        output_size: The size of the output. It should be equal to the number of classes.
        dense_layer_units is the units used in each dense layer
        '''       
        #super().__init__() 
        self.ipt = tf.keras.Input(shape = ipt_shape)
        self.avgpool_layer = GlobalAveragePooling2D()
        self.flatten_layer = Flatten()
        self.fc_layer_1 = Dense(dense_layer_units,activation='relu')
        self.fc_layer_2 = Dense(output_size,activation='softmax')
        
        
        
    def build(self, x, pre_conv = False, pre_pooling = False, 
             attention_num = 3,network_param = [1,1,0],
             resid_params = [[64,3,1,0] for _ in range(3)],
             skip_param = [True, 1]):
        '''
        x: input to ResAttNet model.
        Pre_conv and pre_pooling indicate whether to include the 
        one convolution layer and one pooling layer before starting attention
        module
        
        attention_num: number of attention module in the network

        network_param has three params first is num of residual units in between
        the attention units, second is the extra residual units
        after the last attention units (this number not including the first
        number, say the last attention has 4 resid following it and the number
        in between the two attention module is 1 then the second number should
        be 3) Final number is the number of Dense layers between the global
        average pooling and the final dense softmax activation layers.
        
        Default resid_layer parameter is three layer of 64 layer convolution,
        kernel_size 1, stride 1, padding = same,
        
        skip_param is a tuple, it contain two parameter controlling whether the
        hour glass model has skip connection and how many skip
        connection it have (1 or 2)
        '''
        ipt = x
        skip_connection, skip_mode = skip_param
        if len(network_param) != 3:
            raise Exception('network parameter not of correct length')
        a,b,c = network_param
        if pre_conv:
            x = Conv2D(filters=64,kernel_size=3,strides=1,padding='same')(x)
        if pre_pooling:
            x = AveragePooling2D(padding='same')(x)
        for _ in range(attention_num):
            x = attention_unit(x,resid_layer_params = resid_params,
                    skip_connection=skip_connection,
                    skip_mode=skip_mode)
            for _ in range(a):
                x = residual_units(x,resid_params)

        for _ in range(b):
            x = residual_units(x,resid_params)
        
        x = self.avgpool_layer(x)
        x = self.flatten_layer(x)
        for _ in range(c):
            x = self.fc_layer_1(x)
        out = self.fc_layer_2(x)
        
        return out
