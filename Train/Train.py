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
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")

import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from Train.Loss_Function import Loss_and_Optimizer
from Configuration import Config

variable = Config.Variables()
L_S      = Loss_and_Optimizer()

# %%  Custom Train by Tensor Graph
class Train():
    def __init__(self, train_ds, valid_ds, Generator, Discriminator):

        """
        Custom the GAN-training by TensorFlow Graph
        
        Attributes:
        -----------
            variables:          class type, description in 'Config'
            loss_and_optimizer: class type, include loss function and optimizer
            GAN_Loss:           class type, include GAN loss function
        
            train_ds:           Tensor data,for supervised type training
            valid_ds:           Tensor data,for supervised type training
            
            Generator:          Keras.Model,built-network
            Discriminator:      Keras.Model,built-network
        """
        # Outside <Class> Type
        self.Ls = L_S
        self.Va = variable
        
        # Parameter Definition from <Class> Variables   (called sign: Va)
        self.init_epoch = self.Va.Init_epoch
        self.GAN_epoch  = self.Va.GAN_epoch
        self.Batch_Size = self.Va.batch_size
        
        # Parameter Definition from <Class> GAN_Loss    (called sign: Ls)
        self.G_optimizer_init = tf.keras.optimizers.Adam(0.001)
        self.G_optimizer= self.Ls.G_optimizer
        self.D_optimizer= self.Ls.D_optimizer
        
        # Pixel-based Loss Function
        self.MAE        = self.Ls.MAE
        self.MSE        = self.Ls.MSE
        self.SSIM       = self.Ls.SSIM
        
        # Content-based Loss Function
        if self.Va.channel == 3:
            self.VGG_Loss = self.Ls.VGG_Loss
        elif self.Va.channel == 1:
            self.VGG_Loss = self.Ls.VGG_Loss_Ver2
        
        # Percentage of Loss Function Weights
        self.alpha = self.Ls.Gen_Loss_Percentage
        self.beta  = 1.0
        self.delta = self.Ls.VGG_Loss_Percentage
        self.zeta  = self.Ls.SSIM_Loss_Percentage
        
        # Arrtibutions Definition from Input 
        self.train_ds      = train_ds
        self.valid_ds      = valid_ds
        
        self.G          = Generator
        self.D          = Discriminator
        
        self.v_temp ="Valid -- PSNR:{:.4f}, SSIM:{:.4f}"
    # %% Validation Steps 
    @tf.function
    def test_step(self, val_x, val_y):
        pred_y = self.G(val_x,training=False)
        
        ssim_val_loss= tf.image.ssim(img1 = val_y,img2 = pred_y,max_val=1.006)
        psnr_val_loss= tf.image.psnr(a    = val_y,b    = pred_y,max_val=1.006)
        
        return {
            "v_ssim":ssim_val_loss,
            "v_psnr":psnr_val_loss,
            }
    
    def Print_Valid_Result(self):
        valid_ssim= []
        valid_psnr= []
        
        for val_x,val_y in self.valid_ds:
            valid_loss = self.test_step(val_x = val_x,val_y = val_y)
            
            valid_ssim.append(valid_loss['v_ssim'])
            valid_psnr.append(valid_loss['v_psnr'])
        
        tf.print(self.v_temp.format(
            np.mean(valid_psnr),np.mean(valid_ssim)))
        
    # %% Train Steps: Supervised
    @tf.function
    def Supervised_Train_Step(self,x_train,y_train):
        
        with tf.GradientTape(persistent=True) as tape:
            pred = self.G(x_train,training=True)
            pixel_loss = tf.keras.losses.mse(pred,y_train)
            vgg = self.VGG_Loss(y_true = y_train, y_pred = pred)
                    
            total_loss = (pixel_loss * self.beta +  
                          vgg        * self.delta)
            
        variables = self.G.trainable_variables
        gradients = tape.gradient(total_loss, variables)
        self.G_optimizer.apply_gradients(zip(gradients, variables))
        
        return {
            "pixel_loss":pixel_loss,
            "vgg_loss":vgg,
            }
    
    # %% Train Steps: Unsupervised
    @tf.function
    def Unsupervised_Train_Step(self,x_train,y_train):
        
        with tf.GradientTape(persistent=True) as tape:

            pred = self.G(x_train,training=True)
 
            real_logits = self.D(y_train,training=True)
            fake_logits = self.D(pred,training=True)

            mse  = self.MSE(pred,y_train)
            vgg  = self.VGG_Loss(y_true = y_train,
                                 y_pred = pred)
            
            # Calculate Adversarial Loss 
            D_Loss = self.Ls.CE_D_loss(real_out = real_logits,
                                       fake_out = fake_logits)
            G_Loss = self.Ls.CE_G_loss(real_out = real_logits,
                                       fake_out = fake_logits)
            
            G_total_loss = (G_Loss * self.alpha+
                            mse    * self.beta +
                            vgg    * self.delta)
            
        grad_G = tape.gradient(G_total_loss,
                               self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(grad_G,
                                             self.G.trainable_variables))

        grad_D = tape.gradient(D_Loss,
                               self.D.trainable_variables)
        self.D_optimizer.apply_gradients(zip(grad_D,
                                             self.D.trainable_variables)) 
        return {
            "d_loss":D_Loss,
            "g_loss":G_Loss,
            "mse_loss":mse,
            "vgg_loss":vgg,
            }
        
    # %% Custom Training Loops
    def Adversarial_Training_Ver2(self,train_init=True):
        tf.print(" Adversarial Training Ver2 - Running")
        
        
        # %% Train Loops Supervised
        if train_init:
            tf.print("--"*20)
            tf.print(" Training under Supervised-Type ")
            
            for epoch in range(self.init_epoch):
                progbar = tf.keras.utils.Progbar(
                    int(self.Va.number_of_train_ds/self.Va.batch_size),
                    stateful_metrics=[])
                
                mae_avg = []
                vgg_avg = []
                tf.print("\n Epochs:{}/{}".format(epoch+1,self.init_epoch))
                step = 1

                for x,y in self.train_ds:
                    train_loss = self.Supervised_Train_Step(x_train=x,
                                                            y_train=y)
                    progbar.update(step)
                    step+=1
                    mae_avg.append(train_loss['pixel_loss'])
                    vgg_avg.append(train_loss['vgg_loss'])

                tf.print("Train -- MSE:{:.4f}, VGG:{:.4f}, ".format(
                    np.mean(mae_avg), np.mean(vgg_avg)))
                
                self.Print_Valid_Result()
            # Save the Model
            save_name = '_Pre_Trained_'+str(self.G.name)
            self.G.save(save_name)
            
        # %% Train Loops Unsupervised
        tf.print("--"*20)
        tf.print(" Training under Un-supervised-Type ")
        
        for epoch in range(self.GAN_epoch):
            progbar = tf.keras.utils.Progbar(self.Va.number_of_train_ds,
                                             stateful_metrics=[])
            mae_avg = []
            vgg_avg = []
            d_avg   = []
            g_avg   = []
            tf.print("\n Epochs:{}/{}".format(epoch+1,self.GAN_epoch))
            step = 1
            
            for x,y in self.train_ds:
                train_loss = self.Unsupervised_Train_Step(x_train=x, y_train=y)
                progbar.update(step*self.Va.batch_size)
                step += 1
                mae_avg.append(train_loss['mse_loss'])
                vgg_avg.append(train_loss['vgg_loss'])
                d_avg.append(train_loss['d_loss'])
                g_avg.append(train_loss['g_loss'])
                
            template = "Train -- MSE:{:.4f}, VGG:{:.4f}, D:{:.4f}, G:{:.4f}"
            tf.print(template.format(
                np.mean(mae_avg), np.mean(vgg_avg), 
                np.mean(d_avg)  , np.mean(g_avg)  ,
                ))
            
            self.Print_Valid_Result()
            
        # Save the Model
        current_time= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_name_D = '_GAN_Trained_'+str(self.D.name)+'_'+current_time
        save_name_G = '_GAN_Trained_'+str(self.G.name)+'_'+current_time
        self.D.save(save_name_D)
        self.G.save(save_name_G)
        tf.print("Training End")
        
    # %% Train by Keras API
    def Keras_Fit_Supervised(self,x_train,y_train,x_val,y_val):    
        Networks = self.G
        Networks.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
                         loss      = 'mse',
                         metrics   = ['mse','mae'])
        
        Networks.fit(x = x_train,y = y_train,
                     batch_size = self.Va.batch_size,
                     epochs     = self.Va.Init_epoch,
                     validation_data = (x_val,y_val))
        
        save_name = 'Supervised_Trained_' + str(self.G.name)
        Networks.save(save_name)
        
        
# %%  Custom Train by Keras fit - GAN
class CustomModel(tf.keras.Model):
    def __init__(self,discriminator,generator):
        super(CustomModel,self).__init__()

        self.D = discriminator
        self.G = generator
        self.ls= Loss_and_Optimizer()
        
        if self.ls.MAE_Loss_Percentage == 0:
            self.pixel_loss_name = "MSE"
        elif self.ls.MSE_Loss_Percentage ==0:
            self.pixel_loss_name = "MAE"
        
    def compile(self,
                D_optimizer,
                G_optimizer,
                D_loss,   G_loss, 
                loss_fn):
        
        super(CustomModel,self).compile()
        
        # %% Import the Optimizers
        self.d_optimizer = D_optimizer
        self.g_optimizer = G_optimizer
        
        # %% Import the Loss Function
        self.D_loss = D_loss
        self.G_loss = G_loss
        self.loss_fn= loss_fn
        
    # %% Define the Validation Step
    @tf.function
    def test_step(self,datasets):
        x_valid, y_valid = datasets
        
        y_pred = self.G(x_valid,training=False)

        PSNR     = tf.reduce_mean(tf.image.psnr(
                                    a = y_pred,
                                    b = y_valid,
                                    max_val = 1.012))
        
        SSIM     = tf.reduce_mean(tf.image.ssim(
                                    img1 = y_pred,
                                    img2 = y_valid,
                                    max_val = 1.012))
        
        return {
            "SSIM":SSIM,
            "PSNR":PSNR,
            }
    
    # %% Define the Training Step
    @tf.function
    def train_step(self,datasets):
        x_train,y_train = datasets

        with tf.GradientTape(persistent=True) as tape:
            
            y_pred = self.G(x_train,training=True)
            
            logits_real = self.D(y_train,training=True)
            logits_fake = self.D(y_pred, training=True)
            
            d_loss = self.D_loss(logits_real,logits_fake)
            g_loss = self.G_loss(logits_real,logits_fake)
            
            Loss   = self.loss_fn(y_pred,y_train)
            pixel_loss   = Loss["Pixel_Loss"]
            content_loss = Loss["VGG_Loss"]
            
            t_loss = (
                g_loss       * self.ls.Gen_Loss_Percentage + 
                pixel_loss   * 1.0                         +
                content_loss * self.ls.VGG_Loss_Percentage
                )
            
        D_grads= tape.gradient(d_loss, self.D.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(D_grads,self.D.trainable_variables))
            
        G_grads= tape.gradient(t_loss, self.G.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(G_grads,self.G.trainable_variables))
        
        return {
            self.pixel_loss_name:pixel_loss,
            "VGG":content_loss,
            "d_loss":d_loss,
            "g_loss":g_loss,
            }
    
# %% Custom Train by Keras fit - Supervised        
class Custom_Supervised(tf.keras.Model):
    def __init__(self,generator):
        super(Custom_Supervised,self).__init__()
        
        self.G = generator
        self.ls= Loss_and_Optimizer()
        
        if self.ls.MAE_Loss_Percentage == 0:
            self.pixel_loss_name = "MSE"
        elif self.ls.MSE_Loss_Percentage ==0:
            self.pixel_loss_name = "MAE"
        self.content_loss_name   = "VGG"
        
    def compile(self,optim,loss_fn):
        super(Custom_Supervised,self).compile()
        self.optimizer = optim
        self.loss_func = loss_fn
        
    @tf.function
    def test_step(self,datasets):
        val_x,val_y = datasets
        pred_y = self.G(val_x,training=False)
        
        PSNR     = tf.reduce_mean(tf.image.psnr(
                                    a = pred_y,
                                    b = val_y,
                                    max_val = 1.012))
        
        SSIM     = tf.reduce_mean(tf.image.ssim(
                                    img1 = pred_y,
                                    img2 = val_y,
                                    max_val = 1.012))
        return {
            "SSIM":SSIM,
            "PSNR":PSNR,
            }
    @tf.function
    def train_step(self,datasets):
        train_x,train_y = datasets
        
        with tf.GradientTape() as tape:
            y_pred = self.G(train_x)
            
            Loss = self.loss_func(y_pred,train_y)
            
            pixel_loss   = Loss["Pixel_Loss"]
            content_loss = Loss["VGG_Loss"]
            
            loss = (
                pixel_loss   * 1.0 + 
                content_loss * self.ls.VGG_Loss_Percentage
                )
            
        variables = self.G.trainable_variables
        grads = tape.gradient(loss,variables)
        self.optimizer.apply_gradients(zip(grads,variables))
        
        return {
            self.pixel_loss_name:pixel_loss,
            self.content_loss_name:content_loss,
            }















