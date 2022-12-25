# -*- coding: utf-8 -*-
""" Self-define loss function/object function for network train

Created  Time: Fri Mar 25 15:45:08 2022
Modified Time: Sun May 01 10:32:00 2022 - Ver Modified

@author: JINPENG LIAO

This script include the multi types of the loss function used in 
the image reconstruction task.

This script requires the below packages inside your environment:
    - tensorflow
    - Config

This script can be imported as a module and contains the following class:
    
    * Loss_and_Optimizer <class>:
        include multi-loss function and Adam optimizers.

"""
# %% Import Modules
import os
import sys
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")

import tensorflow as tf
import tensorflow.keras.backend as K
from Configuration import Config

Variables = Config.Variables

# %%  Loss & Optimizer Class
class Loss_and_Optimizer():
    def __init__(self):
        
        """
        Description:
            This class include:
                1. Define the optimizer of the Generator and Discriminator
                2. Define the loss function of the Generator and Discriminator
        """
        self.v = Variables()
        
        self.VGG_19      = self.v.Call_VGG_19('block5_conv4')
        self.VGG_19_B1C2 = self.v.Call_VGG_19('block1_conv2')
        self.VGG_19_B2C2 = self.v.Call_VGG_19('block2_conv2')
        self.VGG_19_B3C4 = self.v.Call_VGG_19('block3_conv4')
        self.VGG_19_B4C4 = self.v.Call_VGG_19('block4_conv4')
        # %% Define the Optimizers
        # Use schedules.LearningRateSchedule to adjust learning rate
        # self.lr_schedule_D = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate = self.v.Discriminator_lr,
        #     decay_steps           = 10000,
        #     decay_rate            = 0.95)
        
        self.lr_schedule_G = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate = self.v.Generator_lr,
            decay_steps           = 10000,
            decay_rate            = 0.95)
        
        self.G_optimizer = tf.keras.optimizers.Adam(self.lr_schedule_G,
                                                    beta_1=self.v.beta1,
                                                    beta_2=self.v.beta2)
        self.D_optimizer = tf.keras.optimizers.Adam(self.v.Discriminator_lr) 
        
        # %% Hyper-Paras for loss percent
        self.Disc_Loss_Percentage= self.v.Loss_Box['disc_loss_percentage']
        
        self.Gen_Loss_Percentage = self.v.Loss_Box['gen_loss_percentage']
        self.MAE_Loss_Percentage = self.v.Loss_Box['mae_loss_percentage']
        self.MSE_Loss_Percentage = self.v.Loss_Box['mse_loss_percentage']
        self.VGG_Loss_Percentage = self.v.Loss_Box['VGG_loss_percentage']
        self.SSIM_Loss_Percentage= self.v.Loss_Box['ssim_loss_percentage']
        
        # %% function for Adversarial
        self.CE = tf.nn.sigmoid_cross_entropy_with_logits
        self.epsilon = 0.000001
        
    # %% Pixel-based Loss
    @tf.function
    def MSE(self,y_true,y_pred):
        " Mean Squared Error Loss Function "
        mse_loss = tf.keras.losses.mse(y_true,y_pred)
        return mse_loss
    @tf.function
    def MAE(self,y_true,y_pred):
        " Mean Absoulty Error Loss Function "
        mae_loss = tf.keras.losses.mae(y_true,y_pred)
        return mae_loss
    @tf.function
    def SSIM(self,y_true,y_pred):
        " Structural Similiarity Loss Function "
        ssim = tf.reduce_mean(tf.image.ssim(img1=y_true,
                                            img2=y_pred,
                                            max_val=1.1))
        return (1.0-ssim)
    # %% Content-based Loss
    @tf.function
    def VGG_Loss(self,y_true,y_pred): # If the VGG-loss require to *255 ?
        """Based on the extracted features from VGG-Network,
        calculate the perceptual loss of two input images"""
        feature_real = self.VGG_19(y_true)
        feature_fake = self.VGG_19(y_pred)
        vgg_loss     = K.mean(K.square(feature_real-feature_fake))
        return vgg_loss
    @tf.function
    def VGG_Loss_Ver2(self,y_true,y_pred):
        """Based on the above VGG loss, can processe the 1-channel
        Tensor dataets, for png upgrade"""
        y_true = tf.keras.layers.Concatenate(axis=-1)([y_true,y_true,y_true])
        y_pred = tf.keras.layers.Concatenate(axis=-1)([y_pred,y_pred,y_pred])
        feature_real = self.VGG_19(y_true)
        feature_fake = self.VGG_19(y_pred)
        # vgg_loss = K.mean(K.square(feature_real-feature_fake))
        vgg_loss = tf.reduce_mean(tf.keras.losses.mse(feature_real,
                                                      feature_fake))
        return vgg_loss
    
    @tf.function
    def VGG_Loss_VerL(self,y_true,y_pred):
        y_true = tf.keras.layers.Concatenate(axis=-1)([y_true,y_true,y_true])
        y_pred = tf.keras.layers.Concatenate(axis=-1)([y_pred,y_pred,y_pred])
        
        real_1 = self.VGG_19(y_true)
        fake_1 = self.VGG_19(y_pred)
        
        real_2 = self.VGG_19_B4C4(y_true)
        fake_2 = self.VGG_19_B4C4(y_pred)
        
        vgg_loss = (tf.reduce_mean(tf.keras.losses.mse(real_1,fake_1))+
                    tf.reduce_mean(tf.keras.losses.mse(real_2,fake_2)))
        return vgg_loss
    
    @tf.function
    def VGG_Loss_VerX(self,y_true,y_pred):
        y_true = tf.keras.layers.Concatenate(axis=-1)([y_true,y_true,y_true])
        y_pred = tf.keras.layers.Concatenate(axis=-1)([y_pred,y_pred,y_pred])
        
        real_1 = self.VGG_19(y_true)
        real_2 = self.VGG_19_B1C2(y_true)
        real_3 = self.VGG_19_B2C2(y_true)
        real_4 = self.VGG_19_B3C4(y_true)
        real_5 = self.VGG_19_B4C4(y_true)

        fake_1 = self.VGG_19(y_pred)
        fake_2 = self.VGG_19_B1C2(y_pred)
        fake_3 = self.VGG_19_B2C2(y_pred)
        fake_4 = self.VGG_19_B3C4(y_pred)
        fake_5 = self.VGG_19_B4C4(y_pred)
        
        l1 = tf.reduce_mean(tf.keras.losses.mse(real_1,fake_1))
        l2 = tf.reduce_mean(tf.keras.losses.mse(real_2,fake_2))
        l3 = tf.reduce_mean(tf.keras.losses.mse(real_3,fake_3))
        l4 = tf.reduce_mean(tf.keras.losses.mse(real_4,fake_4))
        l5 = tf.reduce_mean(tf.keras.losses.mse(real_5,fake_5))
        
        vgg_loss = l1+l2+l3+l4+l5
        return vgg_loss
        
    # %% Adversarial-based Loss
    def RAGAN_D_loss_ver2(self,real_out,fake_out):
        
        real_loss = self.CE(tf.ones_like(real_out),
                            tf.nn.sigmoid(real_out-tf.reduce_mean(fake_out)))
        
        fake_loss = self.CE(tf.zeros_like(fake_out),
                            tf.nn.sigmoid(fake_out-tf.reduce_mean(real_out)))
        
        total_loss= (real_loss+fake_loss)
        return total_loss
    
    def RAGAN_G_loss_ver2(self,real_out,fake_out):

        real_loss = self.CE(tf.zeros_like(real_out),
                            tf.nn.sigmoid(real_out-tf.reduce_mean(fake_out)))
        
        fake_loss = self.CE(tf.ones_like(fake_out),
                            tf.nn.sigmoid(fake_out-tf.reduce_mean(real_out)))
        
        total_loss= (real_loss+fake_loss)
        return total_loss
    
    # self.CE = tf.nn.sigmoid_cross_entropy_with_logits
    @tf.function
    def CE_D_loss(self,real_out,fake_out):
        real_loss = self.CE(tf.ones_like(real_out),real_out)
        fake_loss = self.CE(tf.zeros_like(fake_out),fake_out)
        total_loss= real_loss+fake_loss
        return total_loss
    @tf.function
    def CE_G_loss(self,real_out,fake_out):
        fake_loss = self.CE(tf.ones_like(fake_out),fake_out)
        return fake_loss
    
    """
    Build the RaGAN custom loss by Keras: Power by below websites.
    https://colab.research.google.com/drive/11NlU_Z829NXrHCdWx4ROIIcmfnaxNdR2
    """
    @tf.function
    def RaGAN_D_loss(self,real_out,fake_out):
        real_average_out = K.mean(real_out,axis=0)
        fake_average_out = K.mean(fake_out,axis=0)
        
        Real_Fake_relativistic_average_out = real_out - fake_average_out
        Fake_Real_relativistic_average_out = fake_out - real_average_out
    
        real_loss = K.mean(K.log(K.sigmoid(Real_Fake_relativistic_average_out)+
                                 self.epsilon),axis=0)
        fake_loss = K.mean(K.log(1-K.sigmoid(Fake_Real_relativistic_average_out)+
                                 self.epsilon),axis=0) # Use 0.9 to replace 1.0
        
        D_loss = -(real_loss+fake_loss)
        return D_loss
    @tf.function
    def RaGAN_G_loss(self,real_out,fake_out):
        real_average_out = K.mean(real_out,axis=0)
        fake_average_out = K.mean(fake_out,axis=0)
        
        Real_Fake_relativistic_average_out = real_out - fake_average_out
        Fake_Real_relativistic_average_out = fake_out - real_average_out
        
        real_loss = K.mean(K.log(K.sigmoid(Fake_Real_relativistic_average_out)+
                                 self.epsilon),axis=0)
        fake_loss = K.mean(K.log(1-K.sigmoid(Real_Fake_relativistic_average_out)+
                                 self.epsilon),axis=0)
        
        G_loss = -(real_loss+fake_loss)
        return G_loss


    # %% Pixel & Content Loss
    @tf.function
    def Supervised_Loss(self,y_true,y_pred):
        Pixel_loss = 0
        VGG_loss   = 0
        Total_loss = 0
        
        Pixel_loss = self.Pixel_Loss(y_true,y_pred)
        
        if self.VGG_Loss_Percentage != 0:
            VGG_loss = self.VGG_Loss_Ver2(y_true,y_pred)
        
        return {
            "Pixel_Loss":Pixel_loss,
            "VGG_Loss":VGG_loss,
            }

    @tf.function
    def Pixel_Loss(self,y_true,y_pred):
        Pixel_Loss = 0
        if self.MAE_Loss_Percentage != 0:
            Pixel_Loss = Pixel_Loss + tf.reduce_mean(self.MAE(y_true,y_pred))

        if self.MSE_Loss_Percentage != 0:
            Pixel_Loss = Pixel_Loss + tf.reduce_mean(self.MSE(y_true,y_pred))

        return Pixel_Loss

    @tf.function
    def Supervised_Loss_X(self,y_true,y_pred):
        Pixel_loss = 0
        VGG_loss   = 0
        Total_loss = 0
        
        Pixel_loss = self.Pixel_Loss(y_true,y_pred)
        
        if self.VGG_Loss_Percentage != 0:
            VGG_loss = self.VGG_Loss_VerX(y_true,y_pred)
        
        return {
            "Pixel_Loss":Pixel_loss,
            "VGG_Loss":VGG_loss,
            }

    @tf.function
    def Supervised_Loss_L(self,y_true,y_pred):
        Pixel_loss = 0
        VGG_loss   = 0
        Total_loss = 0
        
        Pixel_loss = self.Pixel_Loss(y_true,y_pred)
        
        if self.VGG_Loss_Percentage != 0:
            VGG_loss = self.VGG_Loss_VerL(y_true,y_pred)
        
        return {
            "Pixel_Loss":Pixel_loss,
            "VGG_Loss":VGG_loss,
            }











