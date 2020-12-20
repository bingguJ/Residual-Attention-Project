import tensorflow as tf

def residual_units(input,layer_params):
# layer_params should be a list containing the convolution layer parameters
# in each list should be organized as
# list(filters,kernel_size,strides,padding(0:same,1:valid)) 
    res_layers = len(layer_params)
    # this one will be the one going through different layer
    cur_input = input
    # a variable storing the original input for later residual addition
    identity = cur_input
    last_filter = 0
    for r in layer_params:
        if len(r) != 4:
            raise Exception('layer parameter not of correct length')
        filters,kernel_size,strides,padding = r
        last_filter = filters
        if padding == 0:
            padding = 'same'
        elif padding == 1:
            padding = 'valid'
# residual unit structured as the full-preactivation in paper "Identity Mapping
# in Deep Residual Networks"
# Batch normalization followed by relu followed by convolution
        cur_input = tf.keras.layers.BatchNormalization()(cur_input)
        cur_input = tf.keras.layers.Activation('relu')(cur_input)
        cur_input = tf.keras.layers.Conv2D(filters=filters,\
                        kernel_size=kernel_size,\
                        strides=strides,padding=padding)(cur_input)

# need to match the shape of the cur_input and the identity so we can add them
# using a convolution with 1*1 filter 
    identity_tensor = tf.keras.layers.Conv2D(filters=last_filter,\
                        kernel_size=1,\
                        strides=1,padding='same')(identity)

    output = tf.keras.layers.Add()([cur_input,identity_tensor])
    return output

def hour_glass_unit(input,r,skip_connection=False,
        skip_mode = 1, layer_params=[[32,3,1,0] for _ in range(3)]):
# this function create the hourglass shaped network in the soft mask branch
# first a maxpool followed by r residual units 
# another maxpool followed by 2r residual units
# upsample then another r residual units
# a final upsample 
# if skip_connection enabled then at the first and second maxpool it branched off going
# through one residual units and added back after the second and first upsample layer
# skip_mode = 1 will only implement the inner skip layer
# skip_mode = 2 will implement both the inner and outer skip layer
# layer_params is the parameter for resid unit, modify it to get residual units
# with different convolution layers
    if not(skip_mode in [1,2]):
        raise Exception('skip mode unknown')
    cur_input = input
    identity0 = input
    cur_input = tf.keras.layers.MaxPool2D(pool_size=(2,2))(cur_input)
    for _ in range(r):
        cur_input = residual_units(cur_input,layer_params)
    
    # save a output for skip connection
    identity = cur_input
    cur_input = tf.keras.layers.MaxPool2D(pool_size=(2,2))(cur_input)
    for _ in range(2*r):
        cur_input = residual_units(cur_input,layer_params)
    
    cur_input = tf.keras.layers.UpSampling2D(size=(2,2),\
            interpolation='bilinear')(cur_input)
    # inner skip connection 
    if skip_connection:
        filters = cur_input.shape[-1]
        identity = residual_units(identity,\
                layer_params)
        cur_input = tf.keras.layers.Add()([cur_input,identity])

    for _ in range(r):
        cur_input = residual_units(cur_input,layer_params)
    filters = cur_input.shape[-1]
    output = tf.keras.layers.UpSampling2D(size=(2,2),\
            interpolation='bilinear')(cur_input)
    
    # outer skip connection
    if skip_connection and skip_mode == 2:
        filters = output.shape[-1]
        identity0 = residual_units(identity0,\
                layer_params)
        output = tf.keras.layers.Add()([output,identity0])

    return output
    

def attention_unit(input,unit_params=(1,2,1),skip_connection=False,skip_mode=1,
        resid_layer_params = [[32,3,1,0] for _ in range(3)]):
# unit parameter is a tuple consist of the p,t,r in the paper 
# Residual Attention network for Image Classification
# attention_unit consist of two branches soft mask branch and trunk branch
    p,t,r = unit_params
    cur_input = input
    for _ in range(p):
        cur_input = residual_units(cur_input,resid_layer_params)
# trunk branch 
# just t residual units
    trunk_input = cur_input
    for _ in range(t):
        trunk_input = residual_units(trunk_input,resid_layer_params)

# soft mask branch
# an hourglass unit followed by 2 convolution with 1*1 kernel
# followed by a sigmoid activation
    cur_input = hour_glass_unit(cur_input,r,skip_connection,skip_mode,resid_layer_params)
# last dimension after the output of residual units
    filters = cur_input.shape[-1]
    cur_input = tf.keras.layers.Conv2D(filters=filters,\
                    kernel_size=1,\
                    strides=1,padding='same')(cur_input)
    cur_input = tf.keras.layers.Conv2D(filters=filters,\
                    kernel_size=1,\
                    strides=1,padding='same')(cur_input)
    sig_activation = tf.keras.layers.Activation('sigmoid')(cur_input)
# assign name to this layer 
    sig_activation = tf.identity(sig_activation, name = 'sigmoid_activation_layer')
    trunk_input = tf.identity(trunk_input, name = 'trunk_layer_output')
# multiplication between soft mask and trunk branch output
# followed by an addition 
# equivalent to the formula on the paper (1+M(x))*T(x)
    cur_input = tf.keras.layers.Multiply()([sig_activation,trunk_input])
    cur_input = tf.keras.layers.Add()([cur_input,trunk_input])
# finally p layers of residual units
    for _ in range(p):
        cur_input = residual_units(cur_input,resid_layer_params)
    
    return cur_input


