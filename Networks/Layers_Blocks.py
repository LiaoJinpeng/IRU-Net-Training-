# -*- coding: utf-8 -*-
""" Self-Define Neural Layers and Neural Blocks for Network Architecture

Created  Time: Fri Mar 25 16:26:29 2022
Modified Time: Sun May 02 08:44:00 2022 - Ver Modified

@author: JINPENG LIAO

This script provide the self-define neural layers and neural blocks, for the 
network architecture design. The initialization of the layer is specific for
the image reconstruction and image denoise.

This script requires the below packages inside your environment:
    - tensorflow

This script can be imported as a module and contains the following class:
    
    * Layers <class>:
        contain the self-define neural layers, power by TensorFlow
    * Blocks <class>:
        contain the self-define neural blocks, power by TensorFlow
    
"""

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")

import tensorflow as tf

class Layers(object):
    def __init__(self):
        self.tmp = 2
        
    def Conv(self,inputs,fs=64,ks=(3,3),s=(1,1),use_bias=False):
        """ 2D Convolution Layer """
        initializer = tf.random_normal_initializer(0., 0.02)
        x1 = tf.keras.layers.Conv2D(filters=fs, kernel_size=ks,
                                    strides=s , padding='same',
                                    use_bias=use_bias,
                                    kernel_initializer=initializer
                                    )(inputs)
        return x1
    
    def Deconv(self,inputs,fs=64,ks=(3,3),s=(1,1),use_bias=False):
        """ 2D DeConvolution Layer """
        initializer = tf.random_normal_initializer(0., 0.02)
        x1 = tf.keras.layers.Conv2DTranspose(filters=fs, kernel_size=ks, 
                                             strides=s , padding='same',
                                             use_bias=use_bias,
                                             kernel_initializer=initializer
                                             )(inputs)
        return x1
    
    def PReLU(self,inputs):
        """ Use Trainable PReLU Activation """
        return tf.keras.layers.PReLU(shared_axes=[1,2])(inputs)
    
    def LeakyReLU(self,inputs):
        """ Use LeakyReLU(0.2) Activation"""
        return tf.keras.layers.LeakyReLU(alpha=0.2)(inputs)
    
    def ReLU(self,inputs):
        return tf.keras.layers.Activation('relu')(inputs)
    
    def ADD(self,inputs,inputs_2):
        """ Add two tensor in axis = -1 """
        return tf.keras.layers.add([inputs,inputs_2])

    def Concat(self,inputs_1,inputs_2,axis=-1):
        """ Concat two tensor in axis = -1 """
        return tf.keras.layers.Concatenate(axis=axis)([inputs_1,inputs_2])

    def Max_Pooling_2D(self,inputs,pooling_size=2):
        """ Max Pooling Layer of Tensor """
        return tf.keras.layers.MaxPooling2D(pool_size = (
            pooling_size, pooling_size))(inputs)

    def BN(self,inputs):
        """ BatchNormalization Layer """
        return tf.keras.layers.BatchNormalization()(inputs)
    
    def Dropout(self,inputs,dropout_rate=0.3):
        """ Dropout Layer """
        return tf.keras.layers.Dropout(rate=dropout_rate)(inputs)
    
class Blocks(Layers):
    def __init__(self):
        super(Blocks,self).__init__()
        self.tmp_2 = 2
    
    def Basic_Residual_Block(self,inputs,filter_num=64,skip_connect=True):
        """ Basic Residual Blocks by He Kai Ming"""
        sc= inputs
        
        x = self.Conv(inputs=inputs,fs=filter_num)
        x = self.BN(x)
        x = self.LeakyReLU(x)
        x = self.Conv(inputs=x,fs=filter_num)
        x = self.BN(x)
        
        if skip_connect:
            x1= self.ADD(x,sc)
        else:
            sc= self.Conv(inputs=sc,fs=filter_num,ks=(1,1))
            x1= self.ADD(x,sc)
            
        return x1
    
    def Down_Residual_Block(self,inputs,filter_num=64):
        """ 2x Downsample by Residual Blocks """
        sc= self.Conv(inputs=inputs,fs=filter_num,ks=(1,1),s=(2,2))
        
        x = self.Conv(inputs=inputs,fs=filter_num)
        x = self.BN(x)
        x = self.LeakyReLU(x)
        
        x = self.Conv(inputs=x,fs=filter_num,s=(2,2))
        x = self.BN(x)
        
        x1= self.ADD(sc,x)
        
        return x1
    
    def dense_block(self,inputs,fs=64,if_T=False,s=2,ks=3):
        def linear_conv(inputs,fs):
            
            x1 = self.Conv(inputs=inputs,fs=fs)
            x1 = self.LeakyReLU(x1)
            
            return x1
        
        x1 = linear_conv(inputs=inputs, fs=fs)
        
        x2 = linear_conv(inputs=x1,fs=fs)
        x2 = self.ADD(x2,x1)
        
        x3 = linear_conv(inputs=x2,fs=fs)
        x3 = self.ADD(x2,x3)
        
        out= tf.keras.layers.concatenate([x1,x2,x3,inputs])
        
        if if_T:
            out = self.Deconv_Block(inputs=out,filter_num=fs)
        else:
            out= self.Conv_Block(inputs     = out,
                                 filter_num = fs,
                                 k_s        = ks,
                                 s          = s)
        
        return out
    
    
    
    # %% Downsample Method
    def Up_Residual_Block(self,inputs,filter_num=64):
        """ Upsample in Residual Method """
        sc= self.Deconv(inputs=inputs,fs=filter_num,ks=(1,1),s=(2,2))
        
        x = self.Deconv(inputs=inputs,fs=filter_num)
        x = self.BN(x)
        x = self.LeakyReLU(x)
        
        x = self.Dropout(x)
        
        x = self.Deconv(inputs=x,fs=filter_num,s=(2,2))
        x = self.BN(x)
        
        x1= self.ADD(x,sc)
        
        return x1

    def Upsamle_2D(self,inputs,filter_num=64,up_size = 2):
        """ Upsample by Interpolation + Conv2D """
        x = tf.keras.layers.UpSampling2D(size=(up_size,up_size),
                                         interpolation='bilinear')(inputs)
        x = self.Conv_Block(inputs=x,filter_num=filter_num)
        return x
    
    def Subpixel_Upsample(self,inputs,filter_num=64):
        """ Upsample by PixelShuffle"""
        x = self.Conv(inputs=inputs,fs=filter_num*4)
        x = tf.nn.depth_to_space(x, block_size=2)
        x = self.BN(x)
        x = self.PReLU(x)
        return x

    # %% Useful Normal Blocks
    def Output_Block(self,inputs,filter_num=64,out_channel=1):
        x = self.Conv_Block(inputs=inputs,filter_num=filter_num)
        
        x = tf.keras.layers.Conv2D(filters     = out_channel,
                                   kernel_size = (3,3),
                                   strides     = (1,1),
                                   activation  = 'tanh',
                                   padding     = 'same')(x)
        return x
    def Conv_Block(self,inputs,filter_num=64,k_s=3,s=1):
        x = self.Conv(inputs=inputs,fs=filter_num,ks=(k_s,k_s),s=(s,s))
        x = self.BN(x)
        x = self.LeakyReLU(x)
        return x
    def Deconv_Block(self,inputs,filter_num=64):
        x = self.Deconv(inputs=inputs,fs=filter_num,s=(2,2))
        x = self.BN(x)
        x = self.LeakyReLU(x)
        return x
    def Input_Block(self,inputs):
        x = self.Conv(inputs=inputs,ks=(7,7))
        x = self.LeakyReLU(x)
        return x
    
    # %%  Blocks for Proposed Ver4 Net
    def RRDB(self,inputs,filter_num=64):
        
        def blocks(inputs,fs=64):
            x = self.Conv(inputs=inputs,fs=fs)
            x = self.PReLU(x)
            return x
        
        sc1 = inputs
        
        x1  = blocks(inputs,fs=filter_num)
        sc2 = x1
        x1  = self.Concat(x1,sc1)
        
        x2  = blocks(x1,fs=filter_num)
        sc3 = x2
        x2  = tf.keras.layers.Concatenate(axis=-1)([x2,sc1,sc2])
        
        x3  = blocks(x2,fs=filter_num)
        
        out = tf.keras.layers.Concatenate(axis=-1)([sc1,sc2,sc3,x3])
        out = self.Conv(inputs=out,fs=filter_num,ks=(3,3))
        out = self.PReLU(out)
        out = self.BN(out)
        
        return out
    
    def Xception_Blocks(self,inputs,filter_num=64):

        DW_1 = self.Conv(inputs=inputs,fs=filter_num,ks=(1,1))
        conv2= self.Conv(inputs=inputs,fs=filter_num,ks=(1,1))
        DW_2 = self.Conv(inputs=conv2, fs=filter_num,ks=(3,3))
        conv3= self.Conv(inputs=inputs,fs=filter_num,ks=(1,1))
        DW_3 = self.Conv(inputs=conv3, fs=filter_num,ks=(5,5))
        
        out = tf.keras.layers.Concatenate(axis=-1)([DW_1,DW_2,DW_3])
        out = self.Conv(inputs=out,fs=filter_num,ks=(3,3))
        out = self.PReLU(out)
        out = self.BN(out)
        
        return out
    
    def Dual_Blocks(self,inputs,filter_num=64,skip_connect=True):
        """Connect Xception Blocks & Residual Dense Block"""
        # Point-wise & Depth-wise Output
        x1 = self.Xception_Blocks(inputs=inputs,filter_num=filter_num)
        # RRDB Output
        x2 = self.RRDB(inputs=inputs,filter_num=filter_num)
        # Concat RRDB and Xception
        out= tf.keras.layers.Concatenate(axis=-1)([x1,x2])
        out= self.Conv(inputs=out,fs=filter_num,
                       ks=(3,3))
        # Skip Output
        if skip_connect:
            x3 = inputs
        else:
            x3 = self.Conv(inputs=inputs,fs=filter_num,
                           ks=(3,3))
        
        # Output Layer
        output = self.ADD(out,x3)
        return output
    
    # %%  Blocks for Proposed Net Ver4_3
    def Depth_Wise_Blocks(self,inputs,filter_num=64):
        DW_1 = self.Conv_Block(inputs=inputs,filter_num=filter_num,
                               k_s=1)
        DW_2 = self.Conv_Block(inputs=inputs,filter_num=filter_num,
                               k_s=3)
        DW_3 = self.Conv_Block(inputs=inputs,filter_num=filter_num,
                               k_s=3)
        
        out  = tf.keras.layers.Concatenate(axis=-1)([DW_1,DW_2,DW_3])
        out  = self.Conv_Block(inputs=out)
        out  = self.ADD(out,inputs)
        return out
    
    def Dense_Blocks(self,inputs,filter_num=64,n_blocks=4):
        concat = inputs
        for _ in range(n_blocks):
            out = self.Conv(inputs=concat,fs=filter_num)
            out = self.LeakyReLU(out)
            concat = self.Concat(concat,out)

        out = self.Conv_Block(inputs=concat,filter_num=filter_num)
        out = self.ADD(out,inputs)
        return out
    
    def Dual_Blocks_Ver2(self,inputs,fs,residual_scaling):
        x1 = self.Dense_Blocks(inputs=inputs,filter_num=fs)
        x2 = self.Depth_Wise_Blocks(inputs=inputs,filter_num=fs)
        out= self.Concat(x1,x2)
        out= self.Conv_Block(out)
        
        out= self.ADD(inputs,out*residual_scaling)
        return out
    # %% Blocks for Compared Used SRResNet 
    def Blocks_for_SRGAN(self,inputs):
        short_cut = inputs
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                                   strides=(1,1), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), 
                                   strides=(1,1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = self.ADD(short_cut,x)
        
        return x    
    
    # %% Blocks for Compared Used ESRGAN
    def residual_block_gen(self,inputs,fs=64,k_s=3,n_blocks=4):
        concat = inputs
        for x in range(n_blocks):
            out = self.Conv(inputs=concat,fs=fs,ks=(k_s,k_s))
            out = tf.keras.layers.PReLU(shared_axes=([1,2]))(out)
            
            concat = tf.keras.layers.concatenate([concat,out])
        out = tf.keras.layers.Conv2D(fs,k_s,padding='same')(concat)
        return out 
    
    
    