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
class Proposed_Networks_Ver2():
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
        self.Gaussian_Noise_r = self.Variables.Gaussian_Noise_Ratio
        
    def Discriminator(self):
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
        
        x = self.Blocks.Input_Block(x)
        x = self.Blocks.Max_Pooling_2D(x)
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs,s=2)
        x = self.Blocks.Max_Pooling_2D(x)
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2)
        x = self.Blocks.Max_Pooling_2D(x)

        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2)
        x = self.Blocks.Max_Pooling_2D(x)

        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=1)
        x = self.Blocks.Max_Pooling_2D(x)

        x = tf.keras.layers.Flatten()(x)
        
        x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
        x = tf.keras.layers.Dense(1)(x)

        model = tf.keras.models.Model(inputs  = Inputs,
                                      outputs = x,
                                      name='Discriminator')
        model.summary()
        return model    
    # %%U-Net+E2E Backbone
    def Proposed_Ver1(self,n_b=2):
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs= 64
        
        x = self.Blocks.Input_Block(Inputs)                      # 512,512,64
        s1= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256*256*128
        s2= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128*128*256
        s3= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64*64*512
        s4= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 32*32*512
        s5= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2)# 16*16*1024
        s6= x 
        
        x1= self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2) #8,8,1024
        for _ in range(n_b):
            x = self.Blocks.Conv_Block(inputs=x1,filter_num=fs*16)
            x1= self.Layers.ADD(x,x1) # Out: 8,8,1024
        
        x = self.Blocks.Deconv_Block(inputs=x1,filter_num=fs*16)# 16,16,1024
        x = self.Layers.Concat(x,s6)            # 16,16,2048
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 32,32,512
        x = self.Layers.Concat(x,s5)            # 32,32,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 64,64,512
        x = self.Layers.Concat(x,s4)            # 64,64,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4)  # 128,128,256
        x = self.Layers.Concat(x,s3)            # 128,128,512
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2)  # 256,256,128 
        x = self.Layers.Concat(s2,x)            # 256,256,256
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1)  # 512,512,64
        x = self.Layers.Concat(s1,x)            # 512,512,128
        
        x = self.Blocks.Output_Block(inputs=x)
        
        ir_model = tf.keras.models.Model(Inputs,x,name='Ver1_Proposed_Net')
        ir_model.summary()
        return ir_model

    # %%U-Net+E2E Backbone
    def Proposed_Ver1_1(self,n_b=2):
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs= 64
        
        x = self.Blocks.Input_Block(Inputs)                      # 512,512,64
        s1= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256*256*128
        s2= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128*128*256
        s3= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64*64*512
        s4= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 32*32*512
        s5= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2)# 16*16*1024
        s6= x 
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2) #8,8,1024
        for _ in range(n_b): 
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*16,
                                                 skip_connect=True)
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*16) # 16,16,1024
        x = self.Layers.Concat(x,s6)            # 16,16,2048
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 32,32,512
        x = self.Layers.Concat(x,s5)            # 32,32,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 64,64,512
        x = self.Layers.Concat(x,s4)            # 64,64,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4)  # 128,128,256
        x = self.Layers.Concat(x,s3)            # 128,128,512
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2)  # 256,256,128 
        x = self.Layers.Concat(s2,x)            # 256,256,256
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1)  # 512,512,64
        x = self.Layers.Concat(s1,x)            # 512,512,128
        
        x = self.Blocks.Output_Block(inputs=x)
        
        ir_model = tf.keras.models.Model(Inputs,x,name='Ver1_Proposed_Net')
        ir_model.summary()
        return ir_model

    # %% U-Net Backbone Stable
    def Proposed_Ver2(self): # Very Stable Version, for future Improvement
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs = 64
        # x = tf.keras.layers.GaussianNoise(stddev=0.4)(Inputs) 
        x = Inputs
        x = self.Blocks.Input_Block(x)
        sc1 = x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2)
        sc2 = x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2)
        sc3 = x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2)
        sc4 = x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2)
        sc5 = x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2)
        sc6 = x
        
        m = self.Layers.Conv(inputs=x,fs=fs*8,s=(2,2))
        m = self.Layers.LeakyReLU(m)
        
        x6 = self.Blocks.Deconv_Block(inputs=m,filter_num=fs*8)
        x6 = self.Layers.Concat(x6,sc6)
        
        x5 = self.Blocks.Deconv_Block(inputs=x6,filter_num=fs*8)
        x5 = self.Layers.Concat(x5,sc5)
        
        x4 = self.Blocks.Deconv_Block(inputs=x5,filter_num=fs*8)
        x4 = self.Layers.Concat(x4,sc4)
        
        x3 = self.Blocks.Deconv_Block(inputs=x4,filter_num=fs*4)
        x3 = self.Layers.Concat(x3,sc3)
        
        x2 = self.Blocks.Deconv_Block(inputs=x3,filter_num=fs*2)
        x2 = self.Layers.Concat(x2,sc2)
        
        x1 = self.Blocks.Deconv_Block(inputs=x2,filter_num=fs*1)
        x1 = self.Layers.Concat(x1,sc1)
        
        x  = self.Blocks.Output_Block(inputs=x1,filter_num=fs)
        
        ir_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                         name='Ver2_0_UNet')
        ir_model.summary()
        return ir_model
    
    # %%U-Net Extrme Net
    def Proposed_Ver3(self,n_b=2):
        """
        A very extrme version based on U-Net backbone
        """
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs= 64
        
        x = self.Blocks.Input_Block(Inputs)                      # 512,512,64
        s1= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256*256*128
        s2= self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*2)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128*128*256
        s3= self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*4)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64*64*512
        s4= self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*8)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 32*32*512
        s5= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2)# 16*16*1024
        s6= x 
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2) #8,8,1024
        for _ in range(n_b): 
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*16,
                                                 skip_connect=True)
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*16) # 16,16,1024
        x = self.Layers.Concat(x,s6)            # 16,16,2048
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 32,32,512
        x = self.Layers.Concat(x,s5)            # 32,32,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 64,64,512
        x = self.Layers.Concat(x,s4)            # 64,64,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4)  # 128,128,256
        x = self.Layers.Concat(x,s3)            # 128,128,512
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2)  # 256,256,128 
        x = self.Layers.Concat(s2,x)            # 256,256,256
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1)  # 512,512,64
        x = self.Layers.Concat(s1,x)            # 512,512,128
        
        x = self.Blocks.Output_Block(inputs=x)
        
        ir_model = tf.keras.models.Model(Inputs,x,name='Ver3_Proposed_Net')
        ir_model.summary()
        return ir_model
    
    def Proposed_Ver3_1(self,n_b=2):
        """
        A very extrme version based on U-Net backbone
        __Use_Residual_Blcoks__
        """
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs= 64
        
        x = self.Blocks.Input_Block(Inputs)                      # 512,512,64
        s1= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256*256*128
        s2= self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*2)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128*128*256
        s3= self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*4)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64*64*512
        s4= self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*8)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 32*32*512
        s5= self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*8)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2)# 16*16*1024
        s6= self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*16) 
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2) #8,8,1024
        for _ in range(n_b): 
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*16,
                                                 skip_connect=True)
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*16) # 16,16,1024
        x = self.Layers.Concat(x,s6)            # 16,16,2048
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 32,32,512
        x = self.Layers.Concat(x,s5)            # 32,32,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 64,64,512
        x = self.Layers.Concat(x,s4)            # 64,64,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4)  # 128,128,256
        x = self.Layers.Concat(x,s3)            # 128,128,512
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2)  # 256,256,128 
        x = self.Layers.Concat(s2,x)            # 256,256,256
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1)  # 512,512,64
        x = self.Layers.Concat(s1,x)            # 512,512,128
        
        x = self.Blocks.Output_Block(inputs=x)
        
        ir_model = tf.keras.models.Model(Inputs,x,name='Ver3_1_Proposed_Net')
        ir_model.summary()
        return ir_model
   
    def Proposed_Ver3_2(self,n_b=2):
        """
        A very extrme version based on U-Net backbone 
        
        Hyper-Paras Setting
        self.beta1 = 0.8                    # Stable Ver: 0.8
        self.beta2 = 0.999
        self.Generator_lr     = 1e-4        # Stable Ver: 1e-4
        self.Discriminator_lr = 2e-4        # Stable Ver: 1e-4
        
        # Training Loss_Func
        self.Loss_Box = {
            'mae_loss_percentage' : 0.00,
            'mse_loss_percentage' : 1.00,
            'ssim_loss_percentage': 0.00,
            'VGG_loss_percentage' : 1e-2,   # Stable Ver: 1e-1
            'gen_loss_percentage' : 1e-3,   # Stable Ver: 1e-3 
                                            # Nightly but work: 1e-2
            'disc_loss_percentage': 1.00,
            }
        
        """
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs= 64

        x = self.Blocks.Input_Block(Inputs)                      # 512,512,64
        s1= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256*256*128
        s2= self.Blocks.dense_block(inputs=x,fs=fs*2,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128*128*256
        s3= self.Blocks.dense_block(inputs=x,fs=fs*4,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64*64*512
        s4= self.Blocks.dense_block(inputs=x,fs=fs*8,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 32*32*512
        s5= self.Blocks.dense_block(inputs=x,fs=fs*8,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2)# 16*16*1024
        s6= self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2) #8,8,1024
        for _ in range(n_b): 
            x = self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*16) # 16,16,1024
        x = self.Layers.Concat(x,s6)            # 16,16,2048
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 32,32,512
        x = self.Layers.Concat(x,s5)            # 32,32,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 64,64,512
        x = self.Layers.Concat(x,s4)            # 64,64,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4)  # 128,128,256
        x = self.Layers.Concat(x,s3)            # 128,128,512
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2)  # 256,256,128 
        x = self.Layers.Concat(s2,x)            # 256,256,256
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1)  # 512,512,64
        x = self.Layers.Concat(s1,x)            # 512,512,128
        
        x = self.Blocks.Output_Block(inputs=x)
        
        ir_model = tf.keras.models.Model(Inputs,x,name='Ver3_2_Proposed_Net')
        ir_model.summary()
        return ir_model    
   
    def Proposed_Ver4(self,n_b=2): # Unstable Version, wait for update
        def dense_for_ver4(inputs,fs):
            x1 = self.Blocks.Conv_Block(inputs=inputs,filter_num=fs)
            x1 = self.Layers.ADD(x1,inputs)
            
            x2 = self.Blocks.Conv_Block(inputs=x1,filter_num=fs)
            x2 = self.Blocks.ADD(x1,x2)
            
            x3 = self.Blocks.Conv_Block(inputs=x2,filter_num=fs)
            x3 = self.Blocks.ADD(x2,x3)
            
            out= tf.keras.layers.concatenate([x1,x2,x3,inputs]) # x,x,c*4
            return out
            
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs= 64
        
        x = self.Blocks.Input_Block(Inputs)                      # 512,512,64
        s1= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256*256*128
        # x = 
        s2= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128*128*256
        # x = 
        s3= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64*64*512
        # x = 
        s4= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 32*32*512
        # x = 
        s5= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2)# 16*16*1024
        # x = 
        s6= x
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2) #8,8,1024
        s7= x
        for _ in range(n_b): 
            x = self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)
        x = self.Layers.Concat(x,s7)            # 8,8,2048
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*16) # 16,16,1024
        x = dense_for_ver4(inputs=x, fs=fs*16)
        x = self.Layers.Concat(x,s6)            # 16,16,2048
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 32,32,512
        x = dense_for_ver4(inputs=x, fs=fs*8)
        x = self.Layers.Concat(x,s5)            # 32,32,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 64,64,512
        x = dense_for_ver4(inputs=x, fs=fs*8)
        x = self.Layers.Concat(x,s4)            # 64,64,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4)  # 128,128,256
        x = dense_for_ver4(inputs=x, fs=fs*4)
        x = self.Layers.Concat(x,s3)            # 128,128,512
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2)  # 256,256,128
        x = dense_for_ver4(inputs=x, fs=fs*2)
        x = self.Layers.Concat(s2,x)            # 256,256,256
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1)  # 512,512,64
        x = self.Layers.Concat(s1,x)            # 512,512,128
        
        x = self.Blocks.Output_Block(inputs=x)
        
        ir_model = tf.keras.models.Model(Inputs,x,name='Ver4_Proposed_Net')
        ir_model.summary()
        return ir_model
   
    def Proposed_Ver5(self,n_b=2):
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs= 64

        x = self.Blocks.Input_Block(Inputs)                      # 512,512,64
        skip_connect = x
        for _ in range(3):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs)
        
        s1= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256*256*128
        s2= self.Blocks.dense_block(inputs=x,fs=fs*2,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128*128*256
        s3= self.Blocks.dense_block(inputs=x,fs=fs*4,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64*64*512
        s4= self.Blocks.dense_block(inputs=x,fs=fs*8,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 32*32*512
        s5= self.Blocks.dense_block(inputs=x,fs=fs*8,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2)# 16*16*1024
        s6= self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2) #8,8,1024
        for _ in range(n_b): 
            x = self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*16) # 16,16,1024
        x = self.Layers.Concat(x,s6)            # 16,16,2048
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 32,32,512
        x = self.Layers.Concat(x,s5)            # 32,32,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 64,64,512
        x = self.Layers.Concat(x,s4)            # 64,64,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4)  # 128,128,256
        x = self.Layers.Concat(x,s3)            # 128,128,512
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2)  # 256,256,128 
        x = self.Layers.Concat(s2,x)            # 256,256,256
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1)  # 512,512,64
        x = self.Layers.Concat(s1,x)            # 512,512,128
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*1)
        for _ in range(3):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs)
        x = self.Layers.ADD(x,skip_connect)
        
        x = self.Blocks.Output_Block(inputs=x)
        
        ir_model = tf.keras.models.Model(Inputs,x,name='Ver5_Proposed_Net')
        ir_model.summary()
        return ir_model    
   
    def Proposed_Ver5_1(self,n_b=2):
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs= 64

        x = self.Blocks.Input_Block(Inputs)                      # 512,512,64
        skip_connect = x
        for _ in range(4):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs)
        
        s1= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256*256*128
        s2= self.Blocks.dense_block(inputs=x,fs=fs*2,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128*128*256
        s3= self.Blocks.dense_block(inputs=x,fs=fs*4,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64*64*512
        s4= self.Blocks.dense_block(inputs=x,fs=fs*8,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 32*32*512
        s5= self.Blocks.dense_block(inputs=x,fs=fs*8,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2)# 16*16*1024
        s6= self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2) #8,8,1024
        for _ in range(n_b): 
            x = self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*16) # 16,16,1024
        x = self.Layers.Concat(x,s6)            # 16,16,2048
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 32,32,512
        x = self.Layers.Concat(x,s5)            # 32,32,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 64,64,512
        x = self.Layers.Concat(x,s4)            # 64,64,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4)  # 128,128,256
        x = self.Layers.Concat(x,s3)            # 128,128,512
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2)  # 256,256,128 
        x = self.Layers.Concat(s2,x)            # 256,256,256
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1)  # 512,512,64
        x = self.Layers.Concat(s1,x)            # 512,512,128
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*1)    # 512,512,64
        for _ in range(2):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs)
        x = self.Layers.Concat(x,skip_connect)
        
        x = self.Blocks.Output_Block(inputs=x)
        
        ir_model = tf.keras.models.Model(Inputs,x,name='Ver5_1_Proposed_Net')
        ir_model.summary()
        return ir_model     
   
    
    def Proposed_Ver5_2(self,n_b=2):
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs= 64

        x = self.Blocks.Input_Block(Inputs)                      # 512,512,64
        skip_connect = x
        for _ in range(4):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs)
        x = self.Layers.ADD(x,skip_connect)
        s1= x
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256*256*128
        s2= self.Blocks.dense_block(inputs=x,fs=fs*2,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128*128*256
        s3= self.Blocks.dense_block(inputs=x,fs=fs*4,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64*64*512
        s4= self.Blocks.dense_block(inputs=x,fs=fs*8,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 32*32*512
        s5= self.Blocks.dense_block(inputs=x,fs=fs*8,s=1)
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2)# 16*16*1024
        s6= self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2) #8,8,1024
        for _ in range(n_b): 
            x = self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*16) # 16,16,1024
        x = self.Layers.Concat(x,s6)            # 16,16,2048
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 32,32,512
        x = self.Layers.Concat(x,s5)            # 32,32,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)  # 64,64,512
        x = self.Layers.Concat(x,s4)            # 64,64,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4)  # 128,128,256
        x = self.Layers.Concat(x,s3)            # 128,128,512
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2)  # 256,256,128 
        x = self.Layers.Concat(s2,x)            # 256,256,256
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1)  # 512,512,64
        x = self.Layers.Concat(s1,x)            # 512,512,128
        
        x = self.Blocks.Output_Block(inputs=x)
        
        ir_model = tf.keras.models.Model(Inputs,x,name='Ver5_1_Proposed_Net')
        ir_model.summary()
        return ir_model     
    
    def Proposed_Ver6(self,n_b=2):
        """
        A very extrme version based on U-Net backbone 
        
        Hyper-Paras Setting
        self.beta1 = 0.8                    # Stable Ver: 0.8
        self.beta2 = 0.999
        self.Generator_lr     = 1e-4        # Stable Ver: 1e-4
        self.Discriminator_lr = 2e-4        # Stable Ver: 1e-4
        
        # Training Loss_Func
        self.Loss_Box = {
            'mae_loss_percentage' : 0.00,
            'mse_loss_percentage' : 1.00,
            'ssim_loss_percentage': 0.00,
            'VGG_loss_percentage' : 1e-2,   # Stable Ver: 1e-1
            'gen_loss_percentage' : 1e-3,   # Stable Ver: 1e-3 
                                            # Nightly but work: 1e-2
            'disc_loss_percentage': 1.00,
            }
        
        """
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        fs= 64

        x = self.Blocks.Input_Block(Inputs)                      # 512,512,64
        s1= x
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*1,s=1) # 512*512*64
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256*256*128
        x = self.Blocks.dense_block(inputs=x,fs=fs*2,s=1)        # 256*256*128
        s2= x
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=1) # 256*256*128
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128*128*256
        x = self.Blocks.dense_block(inputs=x,fs=fs*4,s=1)        # 128*128*256
        s3= x
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=1) # 128*128*256
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64*64*512
        x = self.Blocks.dense_block(inputs=x,fs=fs*8,s=1)        # 64*64*512
        s4= x
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=1) # 64*64*512
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 32*32*512
        x = self.Blocks.dense_block(inputs=x,fs=fs*8,s=1)        # 32*32*512
        s5= x
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=1)# 32*32*512
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2)# 16*16*1024
        x = self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)       # 16*16*1024
        s6= x
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=2) #8*8*1024
        x = self.Blocks.dense_block(inputs=x,fs=fs*16,s=1)        #8*8*1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*16)   # 16,16,1024
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=1) # 16*16*1024
        x = self.Layers.Concat(x,s6)            # 16,16,2048 
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)   # 32*32*512
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=1) # 32*32*512
        x = self.Layers.Concat(x,s5)            # 32,32,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*8)   # 64,64,512
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=1) # 64*64*512
        x = self.Layers.Concat(x,s4)            # 64,64,1024
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4)   # 128,128,256
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=1) # 128,128,256
        x = self.Layers.Concat(x,s3)            # 128,128,512
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2)   # 256,256,128 
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=1) # 256,256,128
        x = self.Layers.Concat(s2,x)            # 256,256,256
        
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1)   # 512,512,64
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*1,s=1) # 512,512,64
        x = self.Layers.Concat(s1,x)            # 512,512,128
        
        x = self.Blocks.Output_Block(inputs=x)
        
        ir_model = tf.keras.models.Model(Inputs,x,name='Ver3_2_Proposed_Net')
        ir_model.summary()
        return ir_model  
   