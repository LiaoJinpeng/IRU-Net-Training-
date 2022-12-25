# -*- coding: utf-8 -*-
""" Basic configration of the Networks Training

Created  Time: Fri Mar 25 15:45:08 2022
Modified Time: Sun May 01 10:32:00 2022 - Ver Modified

@author: JINPENG LIAO

This script allows the user to set the basic Variables and Hyper-Parameters 
of the deep-learning network training.

This script requires the below packages inside your environment:
    - pathlib (for list the filepath)
    - tensorflow (for define the pre-train VGG-19 network)

This script can be imported as a module and contains the following class:
    
    * Variables <class>: 
        config setting of network train
    * Variables.Call_VGG_19 <function>: 
        return a pre-trained VGG-19 network

"""
# %% Import the Essential Modules/Packages
import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pathlib
import tensorflow as tf

# %%  Variables Class Definition
class Variables():
    def __init__(self):
        
        """ 
        A class for definiton of the Parameter using in the Network. This class
        is essential for the deep-learning Networks. The parameters can control
        the initialization of the Project. 
        
        ...
        
        Attributes:
        ----------
        
        Basic Variables of Network Initialization:
        ------------------------------------------
        lr_size : int 
            a int paras to control the input image size. shape=lr_size*lr_size
        hr_size : int
            a int paras to control the output image size.shape=hr_size*hr_size
        channel : int (3 for RGB, 1 for gray)
            a int paras to control the  channel of the image
        Init_epoch : int
            a int paras to control the training epoch of network under the 
            supervised-type training
        GAN_epoch  : int
            a int paras to control the training epoch of network under the
            un-supervised-type training
        batch_size : int
            a int paras to control the batch of datasets in each epochs
        number_of_train_ds : int
            a int paras to control the number of training data sets
        number_of_test_ds  : int
            a int paras to control the number of validation data sets
        continue_Training  : bool
            a bool paras to decide your training based on pre-trained network 
            or start the training based on first-built network.
        -----------------------------------------
        
        Hypers Parameters Tuning:
        -------------------------
        beta1 : float
            beta 1 setting of Adam optimizer
        Generator_lr : float
            a float paras of Geneator/Network learning rate
        Discriminator_lr : float
            a float paras of Discriminator/Network learning rate
        
        Loss_Box : dictionary list
            a dict to control the loss for Generator multi-loss-function
        -------------------------
        
        Datasets File Path Import:
        --------------------------
        Datasets_fp_Box : str
            Control the file path of the train & validation datasets

        """
        # %% Network Basic Variables Definition
        self.ts = time.localtime()
        self.time_date   = str(self.ts[0]) + '.' + str(self.ts[1]) + '.' + \
            str(self.ts[2]) + '_' + str(self.ts[3]) +':' + str(self.ts[4])
            
        # Shape of Datasets
        self.lr_size = 512
        self.hr_size = 512
        self.channel = 1
        
        # Training Epcoh for Init-Generator
        self.Init_epoch = 50
        # Training Epoch for Adversarial train        
        self.GAN_epoch  = 120
        
        # Training Batch Size
        self.batch_size = 4 # was 8
        
        # Training Datasets Amount
        self.number_of_train_ds = 1384
        self.number_of_test_ds  = 400
        
        # Gaussian Noise Prepare for Discriminator
        self.Gaussian_Noise_Ratio = 0.3 # was 0.3 for stable (range: [0.1~0.3])
        # %% Network Hyper-Parameters Tuning
        # Optimizer Learning Rate
        self.beta1 = 0.9                    # Stable Ver: 0.8
        self.beta2 = 0.999
        self.Generator_lr     = 1e-4        # Stable Ver: 1e-4
        self.Discriminator_lr = 2e-4        # Stable Ver: 1e-4
        
        # Training Loss_Func
        self.Loss_Box = {
            'mae_loss_percentage' : 0.00,
            'mse_loss_percentage' : 1.00,
            'ssim_loss_percentage': 0.00,
            'VGG_loss_percentage' : 1e-2,   # Stable Ver: 1e-1/1e-2
            'gen_loss_percentage' : 1e-2,   # Stable Ver: 1e-3 
                                            # Nightly but work: 1e-2
            'disc_loss_percentage': 1.00,
            }
        
        # %% Network Datasets FilePath
        
        self.Datasets_fp_Box = { # For HPC user
            'train_datasets_lr': 
                r'D:\Jinpeng\_nR_Ds\_nR_Datasets\_train\_nR_2',
            'train_datasets_gt': 
                r'D:\Jinpeng\_nR_Ds\_nR_Datasets\_train\_nR_12',
            'validation_datasets_lr' : 
                r'D:\Jinpeng\_nR_Ds\_nR_Datasets\_valid\_nR_2',
            'validation_datasets_gt' : 
                r'D:\Jinpeng\_nR_Ds\_nR_Datasets\_valid\_nR_12',
            }
        
        # self.Datasets_fp_Box = { # For Not HPC user
        #     'train_datasets_lr': 
        #         r'D:\_nR_Datasets\_train\_nR_2',
        #     'train_datasets_gt': 
        #         r'D:\_nR_Datasets\_train\_nR_12',
        #     'validation_datasets_lr' : 
        #         r'D:\_nR_Datasets\_valid\_nR_2',
        #     'validation_datasets_gt' : 
        #         r'D:\_nR_Datasets\_valid\_nR_12',
        #     }
        
            
        self.Data_Argument_Box = {
            'use_RandomFlip':1,
            'use_RandomRotation':1,
            }
        
        self.datasets_GT_fp = pathlib.Path(self.Datasets_fp_Box[
            'train_datasets_gt'])
        self.datasets_LR_fp = pathlib.Path(self.Datasets_fp_Box[
            'train_datasets_lr'])
        self.datasets_VA_fp_lr = pathlib.Path(self.Datasets_fp_Box[
            'validation_datasets_lr'])
        self.datasets_VA_fp_gt = pathlib.Path(self.Datasets_fp_Box[
            'validation_datasets_gt'])

        
    # %% VGG-19 Pre-Trained Network Import
    def Call_VGG_19(self,layers_name = 'block5_conv2'):
        """ A function to extract pre-trained VGG-19 network.
        
        Args:
            None
        
        Returns:
            'keras.models.Model' object, the pre-trained VGG-19
            networks.
        
        """

        # Input the Pre-Trained Model
        VGG_19 = tf.keras.applications.VGG19(
                    weights='imagenet', # Use ImageNet Pre-trained
                    include_top = False,
                    input_shape = (self.hr_size, self.hr_size, 3))
        # Disable the training
        VGG_19.trainable = False
        for l in VGG_19.layers:
            l.trainable  = False
        # Define the Output Layer
        model = tf.keras.models.Model(
            inputs =VGG_19.input,
            outputs=VGG_19.get_layer(layers_name).output
            ) # Previous use : block5_conv4
        #Critical Thinking: if the shallow layer can provide more vessel detail
        return model
    
    # %% Print the Config Setting
    def Print_Config(self):
        template = "\n\
        Input  Image Shape:       {}*{}*{}; \n\
        Ground Truth Image Shape: {}*{}*{}; \n\
        Init Train Epochs:  {}; \n\
        GAN  Train Epochs:  {}; \n\
        Train Batch Size :  {}; \n\
        Number of Train Ds: {}; \n\
        Number of Valid Ds: {}; \n\
                    "
        template_Hyper = "\n\
        Generator     Learning Rate :{:.6f};\n\
        Discriminator Learning Rate :{:.6f};\n\
        Generator beta_1 : {:.4f}; \n\
        Generator beta_2 : {:.4f}; \n\
                          "
        template_Loss  = "\n\
        mae_loss_percentage: {:.5f} ;\n\
        mse_loss_percentage: {:.5f} ;\n\
        vgg_loss_percentage: {:.5f} ;\n\
        gen_loss_percentage: {:.5f} ;\n\
            "
        print("\n")
        print("Network Basic Variables:")
        print(template.format(
            self.lr_size,self.lr_size,self.channel,
            self.hr_size,self.hr_size,self.channel,
            self.Init_epoch,self.GAN_epoch,
            self.batch_size,
            self.number_of_train_ds,self.number_of_test_ds))
        
        print("Hyper Parameters Tuning: ")
        print(template_Hyper.format(self.Generator_lr,
                                    self.Discriminator_lr,
                                    self.beta1,
                                    self.beta2))
        
        print("Network Loss Configuration: ")
        print(template_Loss.format(
            self.Loss_Box['mae_loss_percentage'],
            self.Loss_Box['mse_loss_percentage'],
            self.Loss_Box['VGG_loss_percentage'],
            self.Loss_Box['gen_loss_percentage'],))
        
        
         
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        