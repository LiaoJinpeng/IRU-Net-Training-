# -*- coding: utf-8 -*-
""" Input the Tensor Datasets from the PNG/JPG format files

Created on Fri Mar 25 16:26:29 2022
Modified Time: Sun May 01 11:32:00 2022 - Ver Modified

@author: JINPENG LIAO

This script provide two <class> for tensor datasets convert and input function.
Moreover, it also provide a simple plot function, to plot the processed tensor.

This script requires the below packages inside your environment:
    - tensorflow
    - matplotlib
    - numpy
    
This script requires the below scripts:
    - Config (python script, written by @author)

This script can be imported as a module and contains the following class:
    
    * Data_Loader <class>:
        To import the 8-bits JPG format images and convert to tensor datasets
    * Data_Loader_Png <class>:
        To import the 16-bits PNG format images and convert to tensor datasets
    * Plot <class>:
        Provide a image plot function for preview your network performance
        
"""
# %% Import the Essential Modules/Packages
import os
import sys
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")

import tensorflow    as tf
import numpy         as np
from   matplotlib    import pyplot as plt
from   Configuration import Config

Variables = Config.Variables
# %%  Plot Class Definition
class Plot():
    def __init__(self,test_datasets,original_datasets,model,number_of_image):
        """
        A class to plot the processed Image from the trained network, for the
        previous result pre-view.
        
        Attributes:
        -----------
            test  : Tensor Datasets
                the input datasets of the Network
            train : Tensor Datasets
                the ground-true datasets of the Network
            model : keras.Models object
                the trained Generator Network
            number : int   
                the number of the image, which will be plotted.
        """
        
        self.test  = test_datasets
        self.train = original_datasets
        self.model = model
        self.number= number_of_image
        
    def Image_SR(self,num):
        """ This function aims to return a reconstructed image by Network
        
        Args:
            num: 'int', the number of the processed image
        Return:
            'Tensor object', the processed image
        """
        t1 = time.time()
        img_lr = self.test[num]
        # Reshape: [Weight,Height,Channel] ==> [1,Weight,Height,Channel]
        img_lr = tf.expand_dims(img_lr,axis=0)
        # Image Reconstructed by trained Network
        img_sr = self.model.predict(img_lr)
        # Reshape: [1,Weight,Height,Channel] -> [Weight,Height,Channel]
        img_sr = tf.squeeze(img_sr,axis=0)
        print("--"*30)
        print("Image processing by Generator Cost-Time: {:.3f}s".format(
            time.time()-t1))
        return img_sr
    def Plot(self,obj,head):
        """ This function use the matplotlib.pyplot to plot the result
            
        Args:
            obj:  'Tensor object', the object which will be plotted
            head: 'string', the title of the plotted image
        """
        
        fig = plt.figure()
        plt.imshow(obj)
        plt.title(head)
        plt.axis('off')
        
    def Plot_the_Result(self):
        """ This function use to plot three result: Low-resolution, Original 
            and Super-resolution image

        """
        num = np.random.randint(self.number)
        img_lr = self.test[num]
        img_or = self.train[num]
        img_sr = self.Image_SR(num)
        
        self.Plot(img_lr,'Low-Resolution Image')
        self.Plot(img_or,'Ground-True Image')
        self.Plot(img_sr,'Super-Resolution Image')
        
# %%  Dataset Loader for PNG datasets 
class Data_Loader_Png():
    def __init__(self,
                 if_input_datasets=True,if_test_ds=False):
        """
        A class to convert png datasets to tensor datsets. 
        
        ...
        
        Attributes:
        -----------
        
        if_input_datasets : bool
            if 'True', mean that you want to convert the input input-datasets
            if 'False', mean that you want to convert the ground-truth datasets
        if_test_ds : bool
            if 'True',  mean that you want your datasets as validation datasets
            if 'False', mean that you want your datasets as training   datasets
        
        """
        self.v = Variables()
        self.Channel = self.v.channel
        
        # %% Confirm the file path of target datasets
        if if_test_ds:
            self.Image_Number = self.v.number_of_test_ds
        else:
            self.Image_Number = self.v.number_of_train_ds
        # Decide the Datasets Type
        if if_input_datasets:
            self.ishape    = self.v.lr_size
            if if_test_ds:
                self.data_path = self.v.datasets_VA_fp_lr
            else:
                self.data_path = self.v.datasets_LR_fp
        else: # if not the X, use the Y
            self.ishape    = self.v.hr_size
            if if_test_ds:
                self.data_path = self.v.datasets_VA_fp_gt
            else:
                self.data_path = self.v.datasets_GT_fp

        # %% Obtain the Image Path
        self.image_path      = list(self.data_path.glob('*.png')) 
        self.all_image_paths = [str(path) for path in self.image_path]
        self.image_count     = len(self.all_image_paths)
        
        # %% Preprocess the Image to the Tensor Type
        self.path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        self.image_ds= self.path_ds.map(self.load_and_preprocess_image)
        
        # Using Cache can improve the speed, but it will waste your RAM
        self.using_cache = False
        
    # %% self-functions definition
    def preprocess_image(self,image):
        """
        A function to do the below things:
            1) decode the png format datasets to tensor datasets
            2) resize the tensor to the network input/output shape
            3) normalize the uint16 datasets between [0,1]
        
        Parameters
        ----------
        image : 'string'
            The file name of the png image

        Returns
        -------
        image : 'Tensor'
            'Tensor' object of the converted png image.

        """
        image = tf.image.decode_png(contents = image, 
                                    channels = self.Channel,
                                    dtype    = tf.dtypes.uint16)
        image = tf.image.resize(images = image, 
                                size   = [self.ishape, self.ishape],
                                method = 'bicubic')
        image = image/65535.
        return image
    
    def load_and_preprocess_image(self,path):
        """
        A function to load the png image from the file path and return the 
        processed tensor datasets 

        Parameters
        ----------
        path : 'string'
            The file path of the png format imge

        Returns
        -------
        'Tensor' Object
            The 'Tensor' object of the png image.

        """

        image = tf.io.read_file(path)
        return self.preprocess_image(image)
    
    def return_tensor(self):
        """
        A function to return packaged batch tensor datasets
        """

        if self.using_cache:
            ds = self.image_ds.cache().repeat()
            ds = ds.batch(self.Image_Number)
        else:
            ds = self.image_ds
            ds = ds.repeat()
            ds = ds.batch(self.Image_Number)
        image_batch = next(iter(ds))
        return image_batch
    
    def zip_and_batch_ds(self,inputs_x,inputs_y):
        """
        A function to process the tensor datasets:
            1. zip the inputs_x and inputs_y tensor datasets
            2. batch the zipped datasets 

        Parameters
        ----------
        inputs_x : 'Tensor'
            It should be the input datasets of your network 
        inputs_y : 'Tensor'
            It should be the ground-truth datasets of your network.

        Returns
        -------
        ''Tensor''
            The zipped tensor datasets 

        """
        
        ds = tf.data.Dataset.from_tensor_slices((inputs_x,inputs_y))
        return ds.batch(self.v.batch_size)
    