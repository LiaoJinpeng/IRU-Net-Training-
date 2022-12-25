# -*- coding: utf-8 -*-
""" Proposed Network for Generator and Discriminator

Created  Time: Fri Mar 25 16:26:29 2022
Modified Time: Sun May 02 08:44:00 2022 - Ver Modified

@author: JINPENG LIAO

This script provide the self-define proposed network for training.

This script requires the below packages inside your environment:
    - tensorflow
    - Config        (self-define module in document)
    - Layers_Blocks (self-define module in document)
    
This script can be imported as a module and contains the following class:
    
    * Proposed_Networks <class>:
        Include the proposed Network for training


"""
# %% Modules Import
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")

import time
import tensorflow as tf
from   tensorflow import keras as K

from   Configuration import Config
import Networks.Layers_Blocks as Layers_Blocks

Variables = Config.Variables()
Layers    = Layers_Blocks.Layers()
Blocks    = Layers_Blocks.Blocks()

# %%  Class Network Proposed
class Proposed_Networks():
    def __init__(self):
        """
        Description:
            Networks class is used to build the network
        
        Args:
            channel: the image channel of the input image
            lr_size: the image size of the low-resolution image
            hr_size: the image size of the original image and reconstructed 
                     super-resolution image
            
        Outpus:
            Keras.Model type network
        """
        self.Variables = Variables
        self.Layers    = Layers
        self.Blocks    = Blocks
        
        # %% Import Basic Variables
        self.channel = self.Variables.channel
        self.lr_size = self.Variables.lr_size
        self.hr_size = self.Variables.hr_size
        self.residual_scaling = 0.2
        
        # was 0.3 for stable (range: [0.1~0.3])
        self.Gaussian_Noise_r = self.Variables.Gaussian_Noise_Ratio 
        
        # %% Import Layers
        self.Conv           = self.Layers.Conv
        self.Deconv         = self.Layers.Deconv
        self.PReLU          = self.Layers.PReLU 
        self.LeakyReLU      = self.Layers.LeakyReLU
        self.ReLU           = self.Layers.ReLU
        self.ADD            = self.Layers.ADD
        self.Max_Pooling_2D = self.Layers.Max_Pooling_2D
        self.BN             = self.Layers.BN
        self.Concat         = self.Layers.Concat
        self.Dropout        = self.Layers.Dropout
        
        # %% Import Blocks
        self.RRDB           = self.Blocks.RRDB
        self.Downsampling   = self.Blocks.Down_Residual_Block
        self.Upsampling     = self.Blocks.Up_Residual_Block
        self.Upsamle_2D     = self.Blocks.Upsamle_2D
        self.Dual_Blocks    = self.Blocks.Dual_Blocks 
        
    # %% Discriminator Architecture
    def Discriminator(self):
        Inputs = tf.keras.Input(shape=(self.hr_size,self.hr_size,self.channel))
        
        # Adding Gaussian Noise 
        if self.Gaussian_Noise_r == 0:
            x = Inputs
        else:
            x = tf.keras.layers.GaussianNoise(stddev=self.Gaussian_Noise_r
                                              )(Inputs) 
        
        x = self.Conv(x)
        x = self.LeakyReLU(x)
        
        for i in range(3):
            x = self.Conv(x)
            x = self.Max_Pooling_2D(x)
        
        for i in range(3):
            x = self.Conv(inputs=x,fs=128)
            x = self.Max_Pooling_2D(x)
        
        x = tf.keras.layers.Flatten()(x)
        
        x = tf.keras.layers.Dense(1024)(x)
        x = self.LeakyReLU(x)
        x = self.Dropout(x)
        x = tf.keras.layers.Dense(1,  activation='sigmoid')(x)
        
        model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                      name='Discriminator')
        model.summary()
        return model  
    def Discriminator_Ver2(self):
        """
        Built the Better Classifier for RaGAN training
        """
        fs = 64
        Inputs = tf.keras.Input(shape=(self.hr_size,self.hr_size,self.channel))
        
        # Adding Gaussian Noise 
        if self.Gaussian_Noise_r == 0:
            x = Inputs
        else:
            x = tf.keras.layers.GaussianNoise(stddev=self.Gaussian_Noise_r
                                              )(Inputs) 
        
        x = self.Conv(inputs=x,fs=fs,use_bias=True)
        x = self.LeakyReLU(x)
        x = self.Conv(inputs=x,fs=fs,s=(2,2),use_bias=True)
        x = self.BN(x)
        x = self.LeakyReLU(x)

        x = self.Conv(inputs=x,fs=fs*2,use_bias=True)
        x = self.BN(x)
        x = self.LeakyReLU(x)

        x = self.Conv(inputs=x,fs=fs*2,s=(2,2),use_bias=True)
        x = self.BN(x)
        x = self.LeakyReLU(x)

        x = self.Conv(inputs=x,fs=fs*4,use_bias=True)
        x = self.BN(x)
        x = self.LeakyReLU(x)

        x = self.Conv(inputs=x,fs=fs*4,s=(2,2),use_bias=True)
        x = self.BN(x)
        x = self.LeakyReLU(x)

        x = self.Conv(inputs=x,fs=fs*8,use_bias=True)
        x = self.BN(x)
        x = self.LeakyReLU(x)

        x = self.Conv(inputs=x,fs=fs*8,s=(2,2),use_bias=True)
        x = self.BN(x)
        x = self.LeakyReLU(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024)(x) # Was used 1024 for Proposed Method
        x = self.LeakyReLU(x)
        
        x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

        model = tf.keras.models.Model(inputs  = Inputs,
                                      outputs = x,
                                      name='Discriminator_Ver2')
        model.summary()
        return model
    
    def Discriminator_Ver3(self):
        fs = 64
        Inputs = tf.keras.Input(shape=(self.hr_size,self.hr_size,self.channel))
        
        # Adding Gaussian Noise 
        if self.Gaussian_Noise_r == 0:
            x = Inputs
        else:
            x = tf.keras.layers.GaussianNoise(stddev=self.Gaussian_Noise_r
                                              )(Inputs) 
        x = self.Blocks.Input_Block(x)
        x = self.Blocks.Conv_Block(x,fs)
        
        x = self.Blocks.Conv_Block(x,fs*2,s=2)
        x = self.Blocks.Conv_Block(x,fs*2)
        x = self.Blocks.Conv_Block(x,fs*2)
        
        x = self.Blocks.Max_Pooling_2D(x)
        x = self.Blocks.Conv_Block(x,fs*4)
        x = self.Blocks.Conv_Block(x,fs*4)
        x = self.Blocks.Conv_Block(x,fs*4)
        
        x = self.Blocks.Max_Pooling_2D(x)
        x = self.Blocks.Conv_Block(x,fs*8)
        x = self.Blocks.Conv_Block(x,fs*8)
        x = self.Blocks.Conv_Block(x,fs*8)
        
        x = self.Blocks.Max_Pooling_2D(x)
        x = self.Blocks.Conv_Block(x,fs*8)
        x = self.Blocks.Conv_Block(x,fs*8)
        x = self.Blocks.Conv_Block(x,fs*8)
        
        x = self.Blocks.Max_Pooling_2D(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = self.LeakyReLU(x)
        
        x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

        model = tf.keras.models.Model(inputs  = Inputs,
                                      outputs = x,
                                      name='Discriminator_Ver3')
        model.summary()
        return model
    def Discriminator_Ver4(self): # Based on the Dense_Net121
    
        def dense_layers(inputs,fs):
            x = self.Layers.BN(inputs)
            x = self.Layers.LeakyReLU(x)
            x = self.Layers.Conv(inputs=x,fs=fs,ks=(1,1))
            
            x = self.Layers.BN(inputs)
            x = self.Layers.LeakyReLU(x)
            x = self.Layers.Conv(inputs=x,fs=fs)
            
            out = self.Layers.Concat(x, inputs)
            return out
        
        fs = 64
        Inputs = tf.keras.Input(shape=(self.hr_size,self.hr_size,self.channel))
        # Adding Gaussian Noise 
        if self.Gaussian_Noise_r == 0:
            x = Inputs
        else:
            x = tf.keras.layers.GaussianNoise(stddev=self.Gaussian_Noise_r
                                              )(Inputs) 
            
        x = self.Blocks.Input_Block(inputs=x)
        x1 = self.Layers.Max_Pooling_2D(inputs=x)
        
        for _ in range(5):
            x = dense_layers(inputs=x1,fs=fs)
            x1= self.Layers.Concat(x,x1)
        x = self.Layers.Conv(inputs=x1,fs=fs,ks=(1,1))
        x1= tf.keras.layers.AveragePooling2D(pool_size = (2,2),
                                             strides   = 2,
                                             )(x)
        for _ in range(5):
            x = dense_layers(inputs=x1, fs=fs)
            x1= self.Layers.Concat(x,x1)
        x = self.Layers.Conv(inputs=x1,fs=fs,ks=(1,1))
        x1= tf.keras.layers.AveragePooling2D(pool_size = (2,2),
                                             strides   = 2,
                                             )(x)
        
        for _ in range(5):
            x = dense_layers(inputs=x1, fs=fs)
            x1= self.Layers.Concat(x,x1)
        x = self.Layers.Conv(inputs=x1,fs=fs,ks=(1,1))
        x1= tf.keras.layers.AveragePooling2D(pool_size = (2,2),
                                             strides   = 2,
                                             )(x)
        
        for _ in range(5):
            x = dense_layers(inputs=x1, fs=fs)
            x1= self.Layers.Concat(x,x1)
        x1= tf.keras.layers.AveragePooling2D(pool_size = (2,2),
                                             strides   = 2,
                                             )(x1)
        x1= tf.keras.layers.Flatten()(x1)
        x = tf.keras.layers.Dense(1,activation='sigmoid')(x1)
        model = tf.keras.models.Model(inputs  = Inputs,
                                      outputs = x,
                                      name='Discriminator_Ver4')
        model.summary()
        return model

    # %% Proposed Ver2
    def E2E_U_Net_Novel_Net(self):
        """
        Compared with previous SR_Model_Ver2, the update is:
            1. residual in residual dense block(RRDB) replace the residual block
            2. Delete all BN Layer in the network
            
        """
        
        fs = 64
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = self.Conv(inputs=Inputs,fs=fs,ks=(7,7))
        x = self.PReLU(x)
        
        for i in range(2):
            x1 = self.RRDB(x)
            x1 = self.Conv(inputs=x1,ks=(1,1))
            x  = self.ADD(x,x1*self.residual_scaling)
        
        x_1 = self.Downsampling(x)      # 256
        x_2 = self.RRDB(x_1)            # 256
        x_2 = self.Conv(inputs=x_2,ks=(1,1))
        
        x_3 = self.Downsampling(x_2)    # 128
        x_4 = self.RRDB(x_3)
        x_4 = self.Conv(inputs=x_4,ks=(1,1))
        
        x_5 = self.Downsampling(x_4)    # 64
        x_m = self.RRDB(x_5)            # 64
        x_m = self.Conv(inputs=x_m,ks=(1,1))
        x_m = self.ADD(x_5,x_m)         # 64
        
        x_5_T = self.Upsamle_2D(x_m)        # 128
        x_5_T = self.RRDB(x_5_T)            # 128
        x_5_T = self.Conv(inputs=x_5_T,ks=(1,1))
        x_5_T = self.ADD(x_5_T,x_4)         # 128
        
        x_6_T = self.Upsamle_2D(x_5_T)      # 256
        x_6_T = self.RRDB(x_6_T)            # 256
        x_6_T = self.Conv(inputs=x_6_T,ks=(1,1))
        x_6_T = self.ADD(x_6_T,x_2)         # 256
        
        x_7_T = self.Upsamle_2D(x_6_T)      # 512
        x_7_T = self.RRDB(x_7_T)            # 512
        x_7_T = self.Conv(inputs=x_7_T,ks=(1,1))
        x = self.ADD(x_7_T,x)           # 512
        
        for i in range(4):
            x1 = self.RRDB(x)
            x1 = self.Conv(inputs=x1,ks=(1,1))
            x  = self.ADD(x,x1*self.residual_scaling)
        
        # Output Layer
        x = self.Conv(inputs=x,fs=fs,ks=(3,3))
        x = tf.keras.layers.Conv2D(filters     = self.channel,
                                   kernel_size = (3,3),
                                   strides     = (1,1),
                                   activation  = 'tanh',
                                   padding     = 'same')(x)
        
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='UNet_Ver2_Proposed_Net')
        sisr_model.summary()
        return sisr_model
    
    # %% Proposed Ver4 E2E
    def E2E_Proposed_Net(self):
        
        fs = 64
        n_blocks = 5
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = self.Conv(inputs=Inputs,fs=fs,ks=(7,7))
        x = self.LeakyReLU(x)
        shortcut_1 = x

        x = self.Conv(inputs=Inputs,fs=fs)
        RRDB_Output = self.LeakyReLU(x)        
        
        for _ in range(n_blocks):
            res_output = self.Blocks.Dual_Blocks(inputs = RRDB_Output,
                                                 filter_num = fs)
            res_output = self.BN(res_output) # Using BN, from Ver4_1
            RRDB_Output= self.ADD(RRDB_Output,
                                  res_output*self.residual_scaling)
        
        x = self.Conv(inputs=RRDB_Output,fs=fs)
        x = self.BN(x)
        x = self.LeakyReLU(x)
        x = self.Conv(inputs=x,fs=fs)
        x = self.BN(x)
        x = self.ADD(x,shortcut_1)

        x = self.Conv(inputs=x,fs=fs,ks=(7,7))
        x = tf.keras.layers.Conv2D(filters     = self.channel,
                                   kernel_size = (1,1),
                                   strides     = (1,1),
                                   activation  = 'tanh',
                                   padding     = 'same')(x)
                                   # Use 'tanh' or 'sigmoid'???
                                   # Use the (7,7) kernel to replace (3,3)???
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='E2E_Ver4_Proposed_Network')
        sisr_model.summary()
        return sisr_model
    
    def E2E_Proposed_Net_Ver2(self):
        
        fs = 64
        n_blocks = 4
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = self.Conv(inputs=Inputs,fs=fs,ks=(7,7))
        x = self.PReLU(x)
        shortcut_1 = x

        x = self.Conv(inputs=x,fs=fs)
        x = self.PReLU(x)
        shortcut_2 = x     
        """
        The stable version do not include the 'residual scaling';
        The stable version do not include the shortcut2
        The stable version do not include the ADD function in n_blocks
        """
        for _ in range(n_blocks):
            x1 = self.Blocks.Dual_Blocks(inputs=x,filter_num=fs,
                                         skip_connect=False)
            x = self.BN(x1)
            x = self.ADD(x,x1*self.residual_scaling)
        x = self.ADD(x*self.residual_scaling,shortcut_2)
        
        x = self.Conv(inputs=x,fs=fs)
        x = self.BN(x)
        x = self.PReLU(x)
        
        x = self.Conv(inputs=x,fs=fs)
        x = self.BN(x)
        
        x = self.ADD(x,shortcut_1)

        x = self.Conv(inputs=x,fs=fs)
        x = self.PReLU(x)
        
        x = tf.keras.layers.Conv2D(filters     = self.channel,
                                   kernel_size = (3,3),
                                   strides     = (1,1),
                                   activation  = 'tanh',
                                   padding     = 'same')(x)
                                   # Use 'tanh' or 'sigmoid'???
                                   # Use the (7,7) kernel to replace (3,3)???
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='E2E_Ver4_Proposed_Network_Ver2')
        sisr_model.summary()
        return sisr_model
    
    def E2E_Proposed_Net_Ver3(self):
        fs = 64
        nb = 4
        rs = self.residual_scaling
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = tf.keras.layers.LayerNormalization()(Inputs) # Experiment Used
        x = self.Blocks.Input_Block(x)
        x = self.Blocks.Basic_Residual_Block(x,fs,True)
        sc= x
        
        for _ in range(nb):
            x = self.Blocks.Dual_Blocks_Ver2(x,fs,rs)    
        x = self.ADD(x*rs,sc)
        
        x = self.Blocks.Output_Block(x)
        
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='E2E_Proposed_Net_Ver3')
        sisr_model.summary()
        return sisr_model
    
    
    
    # %% Proposed Ver5
    def U_Net_Proposed_Net(self,if_dropout = True):
        fs = 64
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = self.Conv(inputs=Inputs)
        x = self.LeakyReLU(x)
        
        x = self.Dual_Blocks(x,filter_num=fs*1)             # 512,512,64
        sc1 = self.Dual_Blocks(x,filter_num=fs*1)           # 512,512,64
        
        x = self.Downsampling(inputs=x,filter_num=fs*2)
        x = self.Dual_Blocks(x,filter_num=fs*2)
        sc2 = self.Dual_Blocks(x,filter_num=fs*2)           # 256,256,128
        
        x = self.Downsampling(inputs=x,filter_num=fs*4)
        x = self.Dual_Blocks(x,filter_num=fs*4)
        sc3 = self.Dual_Blocks(x,filter_num=fs*4)           # 128,128,256
        
        x = self.Downsampling(inputs=x,filter_num=fs*8)
        x = self.Dual_Blocks(x,filter_num=fs*8)             # 64,64,512
        
        m = self.Upsampling(inputs=x,filter_num=fs*4)       # 128,128,256
        m = self.Concat(m,sc3)                              # 128,128,512
        
        x = self.Dual_Blocks(m,filter_num=fs*4,skip_connect=False)
        
        x = self.Upsampling(inputs=x,filter_num=fs*2)
        x = self.Concat(x,sc2)
        
        x = self.Dual_Blocks(x,filter_num=fs*2,skip_connect=False)
        x = self.Upsampling(inputs=x,filter_num=fs*1)
        x = self.Concat(x,sc1)
        
        x = self.Dual_Blocks(x,filter_num=fs*1,skip_connect=False)
        
        # Output Layer
        x = self.Conv(inputs=x,fs=fs)
        x = self.LeakyReLU(x)
        x = tf.keras.layers.Conv2D(filters=self.Variables.channel,
                                   kernel_size=(1,1),strides=(1,1),
                                   activation='tanh',padding='same')(x)
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='SR_Model_Ver5_Dual_Block')
        sisr_model.summary()
        return sisr_model














