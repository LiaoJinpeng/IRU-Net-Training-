# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:12:50 2022

@author: JINPENG LIAO

-	Functions:
1.	Define the loss function for network training (Input: Validation&Output)
2.	Define the Optimizer of network training
3.	Define the Network Training Type: 
    a) Supervised 
    b) Semi-supervised , Currently not support
    c) Unsupervised
4.  Define the Network Architecture (Include Return Networks)

"""
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")

import time
import tensorflow as tf
from tensorflow import keras as K

from Configuration import Config
import Networks.Layers_Blocks as Layers_Blocks

Variables = Config.Variables()
Layers    = Layers_Blocks.Layers()
Blocks    = Layers_Blocks.Blocks()

class Comparision_Networks():
    def __init__(self):
        
        self.Variables = Variables
        self.Layers    = Layers
        self.Blocks    = Blocks
        
        self.channel = self.Variables.channel
        self.lr_size = self.Variables.lr_size
        self.hr_size = self.Variables.hr_size
        
        self.Conv    = self.Layers.Conv
        self.Deconv  = self.Layers.Deconv
        self.Concat  = self.Layers.Concat
        self.ADD     = self.Layers.ADD
        
        self.residual_scaling = 0.2
    def DnCNN(self):
        """ Reference:
        Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). 
        Beyond a gaussian denoiser: Residual learning of deep cnn for image 
        denoising. IEEE transactions on image processing, 26(7), 3142-3155.    
        
        """
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
                                   padding='same')(Inputs)
        x = tf.keras.layers.Activation('relu')(x)
        
        for i in range(15):
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),
                                       strides=(1,1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-3)(x)
            x = tf.keras.layers.Activation('relu')(x)  
            
        # Output Layer

        x = tf.keras.layers.Conv2D(filters=self.channel,kernel_size=(3,3),
                                   strides=(1,1), padding='same')(x)
        x = tf.keras.layers.Subtract()([Inputs,x])
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='DnCNN')
        sisr_model.summary()
        return sisr_model 
    
        
    def SR_GAN_Generator(self): # Also the SRResNet
        """ Reference: 
        Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew 
        Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes 
        Totz, Zehan Wang, Wenzhe Shi; 
        Proceedings of the IEEE Conference on Computer Vision and Pattern 
        Recognition (CVPR), 2017, pp. 4681-4690
        """
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        x = self.Conv(inputs=Inputs,ks=(9,9))
        sc= x
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        
        for i in range(5):
            x = self.Blocks.Blocks_for_SRGAN(x)
        
        x = self.Layers.Conv(inputs=x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = self.Layers.ADD(x,sc)
        
        x = self.Layers.Conv(inputs=x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        x = self.Layers.Conv(inputs=x)
        x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
        
        # Output Layer
        x = tf.keras.layers.Conv2D(filters=self.channel,kernel_size=(9,9),
                                   strides=(1,1),
                                   activation='tanh',padding='same')(x)
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='SRResNet')
        sisr_model.summary()
        return sisr_model 
        
    def U_Net(self):
        """ Basic U-Net Architecture, from papers"""
        def blocks(inputs,fs,if_T = False):
            if if_T:
                x = self.Layers.Deconv(inputs=inputs,fs=fs,s=(2,2))
                x = self.Layers.BN(x)
                out = self.Layers.LeakyReLU(x)
            else:
                x = self.Layers.Conv(inputs=inputs,fs=fs,s=(2,2))
                x = self.Layers.BN(x)
                out = self.Layers.LeakyReLU(x)
            return out
        fs = 64
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = self.Conv(inputs=Inputs,fs=fs)        #512,512,64
        sc1 = x
        x = blocks(inputs=x,fs=fs*2,if_T=False)
        sc2 = x
        x = blocks(inputs=x,fs=fs*4,if_T=False)
        sc3 = x
        x = blocks(inputs=x,fs=fs*8,if_T=False)
        sc4 = x
        x = blocks(inputs=x,fs=fs*8,if_T=False)
        sc5 = x
        x = blocks(inputs=x,fs=fs*8,if_T=False)
        sc6 = x
        
        m = self.Layers.Conv(inputs=x,fs=fs*8,s=(2,2))
        m = self.Layers.LeakyReLU(m)
        
        x6 = blocks(inputs=m,fs=fs*8,if_T=True)
        x6 = self.Concat(x6,sc6)
        
        x5 = blocks(inputs=x6,fs=fs*8,if_T=True)
        x5 = self.Concat(x5,sc5)
        
        x4 = blocks(inputs=x5,fs=fs*8,if_T=True)
        x4 = self.Concat(x4,sc4)
        
        x3 = blocks(inputs=x4,fs=fs*4,if_T=True)
        x3 = self.Concat(x3,sc3)
        
        x2 = blocks(inputs=x3,fs=fs*2,if_T=True)
        x2 = self.Concat(x2,sc2)
        
        x1 = blocks(inputs=x2,fs=fs*1,if_T=True)
        x1 = self.Concat(x1,sc1)
        
        x = self.Layers.Conv(inputs=x1,fs=fs)
        x = tf.keras.layers.Activation('relu')(x)
        # Output Layer
        x = tf.keras.layers.Conv2D(filters=self.channel,kernel_size=(3,3),
                                   strides=(1,1),
                                   padding='same')(x)
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='SR_Model_UNet_From_Paper')
        sisr_model.summary()
        return sisr_model 
    
    def DR_U_Net(self):
        
        def blocks(inputs,fs,if_T = False):
            if if_T:
                x = self.Layers.Deconv(inputs=inputs,fs=fs,s=(2,2))
                x = self.Layers.BN(x)
                out = self.Layers.LeakyReLU(x)
            else:
                x = self.Layers.Conv(inputs=inputs,fs=fs,s=(2,2))
                x = self.Layers.BN(x)
                out = self.Layers.LeakyReLU(x)
            return out
        
        fs = 64
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = self.Conv(inputs=Inputs,fs=fs)        #512,512,64
        
        for i in range(4):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*1,
                                                 skip_connect=False)
        x = blocks(inputs=x,fs=fs*2)
        sc1 = x
        
        for i in range(4):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*2,
                                                 skip_connect=False)
        x = blocks(inputs=x,fs=fs*4)
        sc2 = x
        
        for i in range(4):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*4,
                                                 skip_connect=False)
        x = blocks(inputs=x,fs=fs*8)
        sc3 = x
        
        for i in range(4):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*8,
                                                 skip_connect=False)
        
        x = self.ADD(x,sc3)
        x = blocks(inputs=x,fs=fs*4,if_T=True)
        
        for i in range(4):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*4,
                                                 skip_connect=False)
            
        x = self.ADD(x,sc2)
        x = blocks(inputs=x,fs=fs*2,if_T=True)     
        
        for i in range(4):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*2,
                                                 skip_connect=False)
        x = self.ADD(x,sc1)
        x = x = blocks(inputs=x,fs=fs*1,if_T=True)  
        
        for i in range(4):
            x = self.Blocks.Basic_Residual_Block(inputs=x,filter_num=fs*1,
                                                 skip_connect=False)
        
        x = self.Conv(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Output Layer
        x = tf.keras.layers.Conv2D(filters=self.channel,kernel_size=(3,3),
                                   strides=(1,1),
                                   padding='same')(x)
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='SR_Model_DRUNet')
        sisr_model.summary()
        return sisr_model 
        
    def SRDenseNet(self):
        """
        This network is based on the RRDB (Residual in Residual Dense Blocks),
        and the RRDB is first used by DenseNet for Image Classification.
        """
        def Conv_Basic_Block(inputs,filter_size):
            x = self.Layers.Conv(inputs=inputs,fs=filter_size)
            x = self.Layers.PReLU(x)
            x = self.Layers.BN(x)
            return x
        
        fs = 64
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = self.Conv(inputs=Inputs,fs=fs)
        x = self.Layers.PReLU(x)
        sc= x
        x = self.Conv(inputs=x,fs=fs)
        x = self.Layers.PReLU(x)
        x1= self.Layers.BN(x)
        
        for _ in range(5):
            x = self.Blocks.RRDB(inputs=x1,filter_num=fs)
            x1= self.Layers.ADD(x1,x*self.residual_scaling)
        x = self.ADD(x1,sc)
        
        x = Conv_Basic_Block(inputs = x, filter_size = fs)
        x = Conv_Basic_Block(inputs = x, filter_size = fs)
        
        # Output Layer
        x = tf.keras.layers.Conv2D(filters=self.channel,kernel_size=(1,1),
                                   strides=(1,1),
                                   activation='tanh',padding='same')(x)
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='SR_DenseNet')
        sisr_model.summary()
        return sisr_model
            
    def ESRGAN_Generator(self):
        input_lr = tf.keras.layers.Input(shape=(None,None,self.channel))
        
        input_conv = self.Conv(inputs=input_lr,ks=(9,9))
        input_conv = self.Layers.PReLU(input_conv)
        
        ESRRes = input_conv
        for _ in range(5):
            res_output = self.Blocks.residual_block_gen(ESRRes)
            ESRRes = self.ADD(ESRRes,res_output*self.residual_scaling)
        
        ESRRes = self.Conv(inputs=ESRRes)
        ESRRes = self.Layers.BN(ESRRes)
        ESRRes = self.Layers.ADD(ESRRes,input_conv)
        
        output_sr = self.Conv(ESRRes,fs=self.channel,ks=(9,9))
        ESRGAN = tf.keras.models.Model(input_lr,output_sr,
                                       name='ESRGAN_Generator')
        ESRGAN.summary()
        return ESRGAN
        
    def U_Net_format(self):
        """ Basic U-Net Architecture, from papers"""

        fs = 64
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = self.Blocks.Input_Block(Inputs)                         # 512,512,64
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*1,s=1)    # 512,512,64
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*1,s=1)    # 512,512,64
        sc1 = x
        x = self.Blocks.Max_Pooling_2D(inputs=x)                    # 256,256,64
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=1)    # 256,256,128
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=1)    # 256,256,128
        sc2 = x
        x = self.Blocks.Max_Pooling_2D(inputs=x)                    # 128,128,128
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=1)    # 128,128,256
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=1)    # 128,128,256
        sc3 = x
        x = self.Blocks.Max_Pooling_2D(inputs=x)                    # 64,64,256
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=1)    # 64,64,512
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=1)    # 64,64,512
        sc4 = x
        x = self.Blocks.Max_Pooling_2D(inputs=x)                    # 32,32,512
        
        m = self.Blocks.Conv_Block(inputs=x,filter_num=fs*16,s=1)   # 32,32,1024
        m = self.Blocks.Conv_Block(inputs=m,filter_num=fs*16,s=1)   # 32,32,1024
        
        x6 = self.Blocks.Deconv_Block(inputs=m,filter_num=fs*8)     # 64,64,512
        x6 = self.Concat(x6,sc4) # 64,64,512
        x6 = self.Blocks.Conv_Block(inputs=x6,filter_num=fs*8)  # 64,64,512
        x6 = self.Blocks.Conv_Block(inputs=x6,filter_num=fs*8)  # 64,64,512
        
        x5 = self.Blocks.Deconv_Block(inputs=x6,filter_num=fs*4)# 128,128,256
        x5 = self.Concat(x5,sc3) # 128,128,512
        x5 = self.Blocks.Conv_Block(inputs=x5,filter_num=fs*4)  # 128,128,256
        x5 = self.Blocks.Conv_Block(inputs=x5,filter_num=fs*4)  # 128,128,256
        
        x4 = self.Blocks.Deconv_Block(inputs=x5,filter_num=fs*2)# 256,256,128
        x4 = self.Concat(x4,sc2) # 256,256,256
        x4 = self.Blocks.Conv_Block(inputs=x4,filter_num=fs*2)  # 256,256,128
        x4 = self.Blocks.Conv_Block(inputs=x4,filter_num=fs*2)  # 256,256,128
        
        x3 = self.Blocks.Deconv_Block(inputs=x4,filter_num=fs*1)# 512,512,64
        x3 = self.Concat(x3,sc1) # 512,512,128
        x3 = self.Blocks.Conv_Block(inputs=x3,filter_num=fs*1)  # 512,512,64
        x3 = self.Blocks.Conv_Block(inputs=x3,filter_num=fs*1)  # 512,512,64

        # Output Layer
        x = tf.keras.layers.Conv2D(filters=self.channel,kernel_size=(1,1),
                                   strides=(1,1),
                                   padding='same')(x3)
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='U_Net')
        sisr_model.summary()
        return sisr_model         
        
        
    def DRU_Net(self):
        """ Reference:
        Plug-and-Play Image Restoration with Deep Denoiser Prior
        Kai Zhang, Yawei Li, Wangmeng Zuo, Senior Member, IEEE, Lei Zhang , 
        Fellow, IEEE, Luc Van Goal and Radu Timofte, Member, IEEE
        """
        fs = 64
        Inputs = tf.keras.Input(shape=(self.lr_size,self.lr_size,self.channel))
        
        x = self.Blocks.Input_Block(Inputs)
        for _ in range(4):
            x = self.Blocks.Blocks_for_SRGAN(x)
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=2) # 256,256,128
        sc1 = x # 256,256,128
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2,s=1) # 256,256,128
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=2) # 128,128,256
        sc2 = x # 128,128,256
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4,s=1) # 128,128,256
        
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=2) # 64,64,512
        sc3 = x # 64,64,512
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*8,s=1) # 64,64,512
        
        x = self.Blocks.Concat(sc3,x) # 64,64,1024
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*4) # 128,128,256
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*4)   # 128,128,256
        
        x = self.Blocks.Concat(sc2,x) # 128,128,512
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*2) # 256,256,128
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*2)   # 256,256,128
        
        x = self.Blocks.Concat(sc1,x) # 256,256,256
        x = self.Blocks.Deconv_Block(inputs=x,filter_num=fs*1) # 512,512,64
        x = self.Blocks.Conv_Block(inputs=x,filter_num=fs*1)   # 512,512,64
        
        for _ in range(4):
            x = self.Blocks.Blocks_for_SRGAN(x)
        
        # Output Layer
        x = tf.keras.layers.Conv2D(filters=self.channel,kernel_size=(3,3),
                                   strides=(1,1), padding='same')(x)
        # Create Model:
        sisr_model = tf.keras.models.Model(inputs=Inputs,outputs=x,
                                           name='DRU_Net')
        sisr_model.summary()
        return sisr_model 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
