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
import gc
import os
import sys
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("..")

import datetime
import tensorflow as tf
import tensorflow
from tensorflow import keras as K
from Train.Loss_Function import Loss_and_Optimizer
from Configuration import Config
from Networks      import Proposed_Network

variable = Config.Variables()
L_S      = Loss_and_Optimizer()
P_Network= Proposed_Network.Proposed_Networks()
class Train():
    def __init__(self, train_ds, valid_ds):

        """
        Attributes:
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
        self.VGG_Loss   = self.Ls.VGG_Loss
        
        # Percentage of Loss Function Weights
        self.alpha = self.Ls.Gen_Loss_Percentage
        self.beta  = self.Ls.MAE_Loss_Percentage
        self.gamma = self.Ls.MSE_Loss_Percentage
        self.delta = self.Ls.VGG_Loss_Percentage
        self.zeta  = self.Ls.SSIM_Loss_Percentage
        
        # Arrtibutions Definition from Input 
        self.train_ds      = train_ds
        self.valid_ds      = valid_ds
        
        # Metric Box Definition for Training
        self.train_loss_mae = tf.keras.metrics.Mean(name='Train_mae_loss')
        self.train_loss_vgg = tf.keras.metrics.Mean(name='Train_vgg_loss')
        self.train_loss_gen = tf.keras.metrics.Mean(name='Gen_Loss')
        self.train_loss_disc= tf.keras.metrics.Mean(name='Disc_Loss')
        # Metric Box Definition for Validation
        self.vgg_loss  = tf.keras.metrics.Mean(name='VGG_Loss')
        self.ssim_loss = tf.keras.metrics.Mean(name='SSIM_Loss')
        self.mse_loss  = tf.keras.metrics.Mean(name='MSE_Loss')
        self.mae_loss  = tf.keras.metrics.Mean(name='MAE_Loss')
        
        # Record Loss in TensorBoard
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir= 'logs/gradient_tape/'+self.current_time+'/train'
        self.test_log_dir = 'logs/gradient_tape/'+self.current_time+'/test'
        self.train_summary_writer = tf.summary.create_file_writer(
                                    self.train_log_dir)
        self.test_summary_writer  = tf.summary.create_file_writer(
                                    self.test_log_dir)
        
    def Print_Config_Set(self):
        tf.print("**"*30)
        tf.print("Current Generator Name: {}".format(self.G.name))
        tf.print("Optimizer Learning Rate: Gen_Opt:{}; Disc_Opt:{}".format(
            self.Va.Generator_lr,self.Va.Discriminator_lr))
        tf.print("Amount: Train Datasets: {}; VAlidation Datasets: {}".format(
            self.Va.number_of_train_ds,self.Va.number_of_test_ds))
        tf.print("Percentage of Loss:")
        tf.print("MAE:{} ; MSE:{} ; VGG: {} ; G_Loss: {}".format(
            self.beta, self.gamma, self.delta, self.alpha))
        tf.print("Training Epochs: ")
        tf.print("Supervised-Trained:{} ; Unsupervised-Trained:{}".format(
            self.Va.Init_epoch,self.Va.GAN_epoch))
        tf.print("**"*30)
        
    @tf.function
    def Cal_Pixel_and_Content_Loss(self, model, val_x, val_y):
        prediction = model(val_x)
        self.vgg_loss(self.VGG_Loss(y_true=val_y,
                                    y_pred=prediction))
        self.ssim_loss(1.0-self.SSIM(y_true=val_y,
                                     y_pred=prediction))
        self.mse_loss(self.MSE(y_true=val_y,
                               y_pred=prediction))
        self.mae_loss(self.MAE(y_true=val_y,
                               y_pred=prediction))
        
    @tf.function
    def Update_Metrics_State(self):
        self.train_loss_mae.reset_state()
        self.train_loss_vgg.reset_state()
        self.train_loss_gen.reset_state()
        self.train_loss_disc.reset_state()
        
        self.vgg_loss.reset_state()
        self.ssim_loss.reset_state()
        self.mse_loss.reset_state()
        self.mae_loss.reset_state()
        
    def Print_Result_Valid(self):
        tf.print(
            "Valid Loss: VGG:{:.4f};SSIM:{:.4f};MSE:{:.4f};MAE:{:.4f}".format(
            self.vgg_loss.result(),self.ssim_loss.result(),
            self.mse_loss.result(),self.mae_loss.result()))
    
    def Print_Result_Train(self,if_GAN=False):
        if if_GAN:
            template = "Train Loss: MAE:{:.4f};VGG:{:.4f}; D:{:.4f}; G:{:.4f}"
            tf.print(template.format(
                self.train_loss_mae.result(),
                self.train_loss_vgg.result(),
                self.train_loss_disc.result(),
                self.train_loss_gen.result()))
        else:
            template = "Train Loss: MAE:{:.4f};VGG:{:.4f}"
            tf.print(template.format(
                self.train_loss_mae.result(),
                self.train_loss_vgg.result()))
    
    @tf.function
    def Supervised_Train_Step(self,x_train,y_train):
        
        with tf.GradientTape(persistent=True) as tape:
            pred = self.G(x_train,training=True)

            mae = self.Ls.MAE(y_true = y_train,
                              y_pred = pred)
            vgg = self.Ls.VGG_Loss(y_true = y_train,
                                   y_pred = pred)
                    
            total_loss = (mae * self.beta + 
                          vgg * self.delta)
            
            self.train_loss_mae(mae)
            self.train_loss_vgg(vgg)
            
        gradients = tape.gradient(total_loss,
                                  self.G.trainable_weights)
        self.G_optimizer.apply_gradients(zip(gradients,
                                             self.G.trainable_weights))
            
    @tf.function
    def Unsupervised_Train_Step(self,x_train,y_train):
        
        with tf.GradientTape(persistent=True) as tape:
            # Reconstrut Image by Generator
            pred = self.G(x_train,training=True)
            # Output Logits by Discriminator  
            real_logits = self.D(y_train,training=True)
            fake_logits = self.D(pred,training=True)
            # Calculate Pixel Loss
            mae  = self.Ls.MAE(y_true = y_train, # Change to MAE later...
                               y_pred = pred)
            # Calculate VGG Content Loss
            vgg  = self.Ls.VGG_Loss(y_true = y_train,
                                    y_pred = pred)
            # Calculate Adversarial Loss
            D_Loss = self.Ls.RaGAN_D_loss(real_out = real_logits,
                                          fake_out = fake_logits)
            G_Loss = self.Ls.RaGAN_G_loss(real_out = real_logits,
                                          fake_out = fake_logits)
            
            G_total_loss = (G_Loss * self.alpha+
                            mae    * self.beta +
                            vgg    * self.delta)
            # Record the Training Loss
            self.train_loss_mae(mae)
            self.train_loss_vgg(vgg)
            self.train_loss_disc(D_Loss)
            self.train_loss_gen(G_Loss)
            
        grad_G = tape.gradient(G_total_loss,
                               self.G.trainable_weights)
        self.G_optimizer.apply_gradients(zip(grad_G,
                                             self.G.trainable_weights))

        grad_D = tape.gradient(D_Loss,
                               self.D.trainable_weights)
        self.D_optimizer.apply_gradients(zip(grad_D,
                                             self.D.trainable_weights)) 
    
    def Adversarial_Training(self,train_init=True):
        tf.print(" Adversarial Training Ver2 ")
        save_name = '_Pre_Trained_'+str(self.G.name)+'_'+str(self.current_time)
        if train_init:
            
            tf.print("--"*20)
            tf.print(" Training under Supervised-Type ")
            
            for epoch in range(self.init_epoch):
                start_time = time.time()
                for x,y in self.train_ds: # should the init-train use MSE?/MAE?
                    self.Supervised_Train_Step(x_train=x,
                                               y_train=y)
                end_time = time.time()
                
                tf.print("Epochs:{}/{}; --- Time:{:.4f}s".format(
                    epoch+1,self.init_epoch,end_time-start_time))
                self.Print_Result_Train(if_GAN=False)
                
                for val_x,val_y in self.valid_ds:
                    self.Cal_Pixel_and_Content_Loss(model = self.G,
                                                    val_x = val_x,
                                                    val_y = val_y)
                self.Print_Result_Valid()
                self.Update_Metrics_State()
        
        
            self.G.save(save_name)
        
        tf.print("--"*20)
        tf.print(" Training under Un-supervised-Type ")
        save_name_D = '_GAN_Trained_'+str(self.D.name)+'_'+str(self.current_time)
        save_name_G = '_GAN_Trained_'+str(self.G.name)+'_'+str(self.current_time)
        
        for epoch in range(self.GAN_epoch):
            start_time = time.time()
            for x,y in self.train_ds:
                self.Unsupervised_Train_Step(x_train=x,
                                             y_train=y)
            end_time = time.time()

            tf.print("Epochs:{}/{}; Cost Time:{:.4f}s".format(
                epoch+1,self.GAN_epoch,end_time-start_time))
            self.Print_Result_Train(if_GAN=True)
            
            for val_x,val_y in self.valid_ds:
                self.Cal_Pixel_and_Content_Loss(model = self.G,
                                                val_x = val_x,
                                                val_y = val_y)
            self.Print_Result_Valid()
            self.Update_Metrics_State()
            
        self.D.save(save_name_D)
        self.G.save(save_name_G)
        tf.print("Training End")
           
    def Keras_Fit_Supervised(self,x_train,y_train,x_val,y_val):    
        Networks = self.G
        Networks.compile(optimizer = tf.keras.optimizers.Adam(1e-4),
                         loss      = 'mae',
                         metrics   = ['mse','mae'])
        
        Networks.fit(x = x_train,y = y_train,
                     batch_size = self.Va.batch_size,
                     epochs     = self.Va.Init_epoch,
                     validation_data = (x_val,y_val))
        
        save_name = 'Supervised_Trained_' + str(self.G.name)
        Networks.save(save_name)
    
    def Compute_G_C_Loss(self,y_true,y_pred):
        mae = self.Ls.MAE(y_true,y_pred)
        vgg = self.Ls.VGG_Loss(y_true,y_pred)
        loss= mae+vgg
        return tf.nn.compute_average_loss(loss,
                                          global_batch_size=self.Va.batch_size)
    
    def Distribute_Unsupervised_train_step(self,inputs_ds):
        pass
    
    def Distribute_Adversarial_Train(self):
        gc.collect()
        strategy = tensorflow.distribute.MirroredStrategy()
        with strategy.scope():
            distr_train_ds = strategy.experimental_distribute_dataset(
                self.train_ds)
            distr_valid_ds = strategy.experimental_distribute_dataset(
                self.valid_ds)
            
            Generator     = P_Network.E2E_Proposed_Net_Ver2()
            Discriminator = P_Network.Discriminator_Ver2()
            
            G_optimizer   = tf.keras.optimizers.Adam(0.0001)
            D_optimizer   = tf.keras.optimizers.Adam(0.0005)
            
            train_mae_metrics = tf.keras.metrics.Mean()
            valid_mae_metrics = tf.keras.metrics.Mean()

            def Adversarial_G_loss(real_out,fake_out):
                G_Loss = self.Ls.RaGAN_G_loss(real_out,fake_out)
                return G_Loss
            def Compile_G_loss(y_true,y_pred):
                mae = self.Ls.MAE(y_true,y_pred)
                vgg = self.Ls.VGG_Loss(y_true,y_pred)
                loss= mae+vgg
                return loss
            def Adversarial_D_loss(real_out,fake_out):
                D_Loss = self.Ls.RaGAN_D_loss(real_out,fake_out)
                return D_Loss
            
            def train_step(datasets):
                x_train,y_train = datasets
                with tf.GradientTape(persistent=True) as tape:
                    
                    pred = Generator(x_train,training=True)
                    
                    logits_real = Discriminator(y_train,training=True)
                    logits_fake = Discriminator(pred,training=True)
                    
                    G_loss = Adversarial_G_loss(logits_real, logits_fake)
                    pixel_l= Compile_G_loss(y_train, pred)
                    
                    D_loss = Adversarial_D_loss(logits_real, logits_fake)
                    G_loss = 0.001*G_loss + pixel_l
                    
                G_Var = Generator.trainable_variables
                D_Var = Discriminator.trainable_variables
                    
                Grads_G = tape.gradient(G_loss, G_Var)
                G_optimizer.apply_gradients(zip(Grads_G,G_Var))
                    
                Grads_D = tape.gradient(D_loss, D_Var)
                D_optimizer.apply_gradients(zip(Grads_D,D_Var))
                    
                train_mae_metrics.update_state(pixel_l)
                
                return G_loss,D_loss
            
            def valid_step(datasets):
                x_valid,y_valid = datasets
                pred = Generator(x_valid,training=False)
                pixel_l = Compile_G_loss(y_valid, pred)
                valid_mae_metrics.update_state(pixel_l)
                    
            @tf.function
            def distributed_training(datasets):
                epoch = 0
                for Ds in distr_train_ds:
                    epoch += 1
                    strategy.run(train_step,args=(Ds,))
                    # strategy.experimental_run_v2()
                    tf.print('Step:',epoch,';Loss:',train_mae_metrics.result())
                    gc.collect()
            @tf.function
            def distributed_valid(datasets):
                for Ds in distr_valid_ds:
                    strategy.experimental_run_v2(valid_step,args=(Ds,))
                    gc.collect()
                    
        for epoch in range(self.GAN_epoch):
            distributed_training(distr_train_ds)
            gc.collect()
            train_mae_metrics.reset_states()
            valid_mae_metrics.reset_states()
            
            






















