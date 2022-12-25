# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 13:24:10 2022

@author: JINPENG LIAO

-	Functions:

    "Based on USER, to use above four files to individually training your 
    personal-define networks."
    
    1. Insert the Datasets
    2. Instantiate the Objects
"""
# %% Import the Essential Modules/Packages
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append("..")

from Configuration   import Config
from Configuration   import Datasets_Loader

from Train.Train     import Train as Train_class
from Train.Train     import CustomModel
from Train.Train     import Custom_Supervised
from Train           import Loss_Function

from Networks        import Proposed_Network
from Networks        import Proposed_Network_Ver2
from Networks        import Layers_Blocks
from Networks        import Compared_Network

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
# %% Instantiate Object
v           = Config.Variables
variable    = Config.Variables()
ls          = Loss_Function.Loss_and_Optimizer()

# if use 3-channel jpg as input, use the <class>: Datasets_Loader.Data_Loader
Data_loader = Datasets_Loader.Data_Loader_Png 
Plot_Class  = Datasets_Loader.Plot

Proposed_Net_Ver2 = Proposed_Network_Ver2.Proposed_Networks_Ver2()
Proposed_Net= Proposed_Network.Proposed_Networks()
Compared_Net= Compared_Network.Comparision_Networks()

# %% Import Network
C_train = True
if C_train:
    Generator     = tf.keras.models.load_model(
        '_Supervised_Trained_Ver3_2_Proposed_Net_a1_b001_ALL_Out_',
                                               compile=False)
    
    # Discriminator = Proposed_Net_Ver2.Discriminator()
    # Discriminator = tf.keras.models.load_model(Discriminator_file,
    #                                            compile=False)
else:
    Generator     = Proposed_Net_Ver2.Proposed_Ver3_2()
    # Discriminator = Proposed_Net_Ver2.Discriminator()
    
# %% Import Dataset  
# Train Datasets for supervised 
x_train = Data_loader(if_input_datasets = True,
                      if_test_ds        = False).return_tensor()
x_train = tf.keras.layers.GaussianNoise(stddev=0.4)(x_train) 

y_train = Data_loader(if_input_datasets = False,
                      if_test_ds        = False).return_tensor()
# Validation Datasets for Model Testing
x_test  = Data_loader(if_input_datasets = True,
                      if_test_ds        = True).return_tensor()
y_test  = Data_loader(if_input_datasets = False,
                      if_test_ds        = True).return_tensor()

# Train Datasets for unsupervised GAN
train_ds = Data_loader().zip_and_batch_ds(inputs_x = x_train,
                                          inputs_y = y_train)
# Validation Datasets for unsupervised GAN
valid_ds = Data_loader().zip_and_batch_ds(inputs_x = x_test,
                                          inputs_y = y_test)

# Experimental Method:
train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
valid_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# %% Realize the Train Class in <Module> Train

""" Stable Ver for Train Class"""
# Train_C = Train_class(train_ds = train_ds, 
#                       valid_ds = valid_ds,
#                       Generator     = Generator,
#                       Discriminator = Discriminator)


# %% Run the Main Functions/Progress
if __name__ == "__main__":
    tf.print("Begin Time: {}".format(variable.time_date))
    tf.print("Generator : {}".format(Generator.name))
    
    variable.Print_Config()
    
    #%%  Method Used Custom Method //UNSTABLE VERSION//
    # Train_C.Adversarial_Training_Ver2(train_init=True)
    
    #%%  Method Used Simplest Keras Method 
    
    # Generator.compile(optimizer = tf.keras.optimizers.Adam(1e-6),
    #                   loss      = 'mse')
    # hist = Generator.fit(x = x_train,
    #                      y = y_train,
    #                      batch_size = variable.batch_size,
    #                      epochs     = variable.Init_epoch,
    #                      validation_data = (x_test,y_test))
    
    # save_name = '_Supervised_Trained_' + str(Generator.name) +'_MSE_ONLY_Ver_CTrain'
    # Generator.save(save_name)
    
    
    #%% Method used Keras Custom Fit and Compile

    # %% Train under Supervised
    gen = Custom_Supervised(generator = Generator)
    gen.compile(optim   = ls.G_optimizer,
                loss_fn = ls.Supervised_Loss_X)
    hist_supervised = gen.fit(train_ds, epochs = variable.Init_epoch, 
                              validation_data  = valid_ds)
    gen.G.save("_Supervised_Trained_"+str(Generator.name)+'_a1_b001_ALL_Out'+"_Ver_CTrain_")
    # Generator = gen.G
    # Optimizer_pre = gen.optimizer
    
    # %% Train under Un-supervised
    
    # irgan = CustomModel(discriminator = Discriminator,
    #                     generator     = Generator
    #                     )
        
    # irgan.compile(D_optimizer = ls.D_optimizer,
    #               G_optimizer = ls.G_optimizer, #'Optimizer_pre' when use above
    #               D_loss      = ls.RAGAN_D_loss_ver2,
    #               G_loss      = ls.RAGAN_G_loss_ver2,
    #               loss_fn     = ls.Supervised_Loss
    #               )
        
    # history = irgan.fit(train_ds,
    #               epochs = variable.GAN_epoch,
    #               validation_data = valid_ds)
    
    # irgan.G.save('_GAN_Trained_Ver3_2_GaussianNoise_Ver_New')
    # irgan.D.save('_GAN_Trained_Proposed_Net_D_')
    
    
    
    
    
    
    