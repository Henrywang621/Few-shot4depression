import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.activations import softmax, relu
import os
import numpy as np
import time
from tensorflow.keras import regularizers
tf.keras.backend.set_floatx('float32')

class EEGFSNet(tf.keras.Model):

    def __init__(self,
                dims_inf = [32,32,32],
                dims_pred = [32,32,32],
                activation = "relu",
                time=3000,
                nchan=68,
                batchnorm = False,
                block = "conv",
                merge = False,
                dropout = 0.0):
        
        super(EEGFSNet, self).__init__()
        self.enc_type  = "None"

        if len(block) == 1:
            block = f"{block},{block},{block},{block}"

        self.block = block

        # # Prediction network
        self.dense_fz = getSequential(dims=dims_pred,activation=activation,final=True,name="pred_dense_fz")


        self.time_fz = getTimeBlock(block=block[-1],dims=dims_pred,activation=activation,final=False,name="pred_time_fz",input_shape=(time,dims_inf[-1]+1),batchnorm=batchnorm)

        self.chans_fz = getTimeBlock(block=block[-2],isSPfilters=True,nchan=nchan,dims=dims_pred[0],activation=activation,final=False,name="pred_SPfilters_fz",input_shape=(time,dims_inf[-1]+1),batchnorm=batchnorm)

        # self.time_gz = getTimeBlock(block=block[-1],dims=dims_pred,activation=activation,final=False,name="pred_time_gz",input_shape=(time,dims_pred[-1]),batchnorm=batchnorm)


        # # Support and Query network (start with both same weights)
        self.time_fv = getTimeBlock(block=block[2],dims=dims_inf[0],kernel_size=32,activation=activation,final=False,name="s_time_fv",input_shape=(time,dims_inf[-1]+1),batchnorm=batchnorm)
        self.time_gv = getTimeBlock(block=block[2],dims=dims_inf[0],kernel_size=16,activation=activation,final=False,name="s_time_gv",input_shape=(time,dims_inf[-1]),batchnorm=batchnorm)
        self.dense_fv = getSequential(dims=dims_inf[:2],activation=activation,final=False,name="s_dense_fv")
        self.dense_fw = getSequential(dims=dims_inf[:2],activation=activation,final=False,name="s_dense_fw")



        self.time_uf = getTimeBlock(block=block[1],dims=dims_inf[0],kernel_size=64,activation=activation,name="ux_time_f",input_shape=(time,dims_inf[-1]+1),batchnorm=batchnorm)
        self.chans_ufg = getTimeBlock(block=block[-2],isSPfilters=True,nchan=nchan,dims=dims_pred[0],activation=activation,final=False,name="pred_spfilters_fg",input_shape=(time,dims_inf[-1]+1),batchnorm=batchnorm)
        self.time_v  = getTimeBlock(block=block[0],dims=dims_inf,activation=activation,final=False,name="vb_time_v",input_shape=(time,1),batchnorm=batchnorm)

        self.dense_v  = getSequential(dims=dims_inf,activation=activation,final=False,name="vb_dense_v")
        self.time_uz = getTimeBlock(block=block[1],dims=dims_inf[0],kernel_size=64,activation=activation,name="ux_time_z",input_shape=(time,2*dims_inf[-1]+1),batchnorm=batchnorm)

        


    def call(self, inp, training=False):
        que_x, sup_x, sup_y = inp
        sup_y = tf.cast(sup_y, tf.float32)

        M = tf.shape(sup_x)[0] # Metabatch
        N = tf.shape(sup_x)[1] # Batch
        T = tf.shape(sup_x)[2] # Time
        F = tf.shape(sup_x)[3] # Channels/Features


        zero_count = tf.reduce_sum(tf.concat([que_x,sup_x],axis=1),axis=[1,3])
        zero_count = tf.math.count_nonzero(zero_count,axis=1,dtype=tf.dtypes.float32)
        zero_count = tf.expand_dims(zero_count,-1)
        zero_count = tf.expand_dims(zero_count,-1)
        


        # Encode sup_x MxNxTxF to MxFxTxK 
        sup_x_1 = tf.transpose(sup_x,[0,1,3,2])  # MxNxFxT
        vs_bar = tf.expand_dims(sup_x_1,-1)      # MxNxFxTx1
        vs_bar = self.time_v(vs_bar,training)            # MxNxFxTxK
        


        # Encode sup_y MxNx1 to Mx1xK
        sup_y1 = tf.reduce_mean(sup_y, axis=-1) #MxN
        sup_y1 = tf.expand_dims(sup_y1,axis=-1)  # MxNx1
        cs_bar = tf.expand_dims(sup_y1,axis=-1)  # MxNx1x1    
        cs_bar = self.dense_v(cs_bar)            # MxNx1xK 
 
        
        ##### U network #####  (DS over Channels)

        sup_x_1 = tf.expand_dims(sup_x_1,axis=-1) # MxNxFxTx1
        u_xs = tf.concat([sup_x_1,vs_bar],-1) # MxNxFxTx(K+1)

        u_xs = self.time_uf(u_xs,training) # MxNxFxTxK

        u_xs=tf.transpose(u_xs, [0,1,3,2,4]) # MxNxTxFxK

        u_xs = self.chans_ufg(u_xs, training) #MxNxTx1xK

        u_xs = tf.squeeze(u_xs, axis=-2) #MxNxTxK

        u_ys = tf.tile(cs_bar,[1,1,T,1]) # MxNxTxK 


        u_s  = u_xs + u_ys # MxNxTxK 



        in_xs = tf.tile(tf.expand_dims(u_s,axis=3),[1,1,1,F,1]) # MxNxTxFxK
        in_xs = tf.transpose(in_xs, [0,1,3,2,4]) # MxNxFxTxK)


        in_ys = tf.reduce_mean(u_ys,axis=2)    # MxNxK

        in_ys = tf.concat([sup_y1,in_ys],axis=-1) # MxNx(K+C)

        in_ys = self.dense_fv(in_ys)        # MxNxK
        in_ys = tf.reduce_mean(in_ys,axis=1) # MxK
        in_ys = self.dense_fw(in_ys)        # MxK
        in_ys = tf.tile(tf.expand_dims(in_ys,axis=1),[1,N,1]) # MxNxK


       
        in_xs = self.time_fv(in_xs,training) # MxNxFxTxK
        in_xs = tf.reduce_mean(in_xs, axis=1) # MxFxTxK
        in_xs = self.time_gv(in_xs,training)     # MxFxTxK
        in_xs = tf.transpose(in_xs,[0,2,1,3]) # MxTxFxK

        p_xs = tf.tile(tf.expand_dims(in_xs, axis=1),[1,N,1,1,1]) # MxNxTxFxK
        que_x_1 = tf.expand_dims(que_x, axis=-1) # MxNxTxFx1    

        z = tf.concat([p_xs,que_x_1],axis=-1) # MxNxTxFx(K+1)
            
        z = tf.transpose(z,[0,1,3,2,4]) # MxNxFxTx(K+1)

        zg = self.time_fz(z,training) # MxNxFxTxK

        z = tf.concat([z,zg],-1) # MxNxFxTx(2K+1)

        z = self.time_uz(z,training) # MxNxFxTxK

        # (Ds over Channels)
        
        z = tf.transpose(z, [0,1,3,2,4]) # MxNxTxFxK

        z = self.chans_fz(z, training) #MxNxTx1xK

        z = tf.squeeze(z, axis=-2) #MxNxTxK

        out = tf.reduce_mean(z, axis=2) # MxNxK


        out = tf.concat([out,in_ys],-1) # MxNx(K+K)
        out = self.dense_fz(out)

        return out

def getTimeBlock(block = "conv",dims=[32,32,1],isSPfilters=False,nchan=68,input_shape=None,kernel_size=None,activation=None,name=None,final=None,batchnorm=False,dilate=False):

    if block == "conv":

        return convBlock(dims=dims,input_shape=input_shape,kernel_size=kernel_size,activation=activation,name=name,batchnorm=batchnorm,dilate=dilate)
    
    elif block == "convSP":

        return convBlock(dims=dims,isSPfilters=True,nchan=nchan,input_shape=input_shape,kernel_size=kernel_size,activation=activation,name=name,batchnorm=batchnorm,dilate=dilate)
    

    elif block == "gru":

        return gruBlock(dims=dims,input_shape=input_shape,activation=activation,name=name,final=final)

    else:
        raise ValueError(f"Block type {block} not defined.")


class convBlock(tf.keras.Model):

    def __init__(self,dims=32,isSPfilters=False,nchan=None,input_shape=None,kernel_size=None,activation=None,name=None,final=True,batchnorm=False,dilate=False):
        
        super(convBlock, self).__init__()

        self.batchnorm = batchnorm
        self.final = final
        dilation = [1,1,1]
        
        if isSPfilters==False:
            self.c1 = tf.keras.layers.Conv1D(filters=dims,kernel_size=kernel_size, activation=None,name=f"{name}-0",padding="same",dilation_rate=dilation[0],input_shape=input_shape)
            self.relu1 = tf.keras.layers.Activation(activation)
        else:
            self.c1 = tf.keras.layers.Conv2D(dims, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1=0), padding="valid")
            self.relu1 = tf.keras.layers.Activation(activation)
        if self.batchnorm:
            self.bn1 = tf.keras.layers.BatchNormalization()                
        

    def call(self, inp, training=False):
        
        out = self.c1(inp)
        if self.batchnorm:
            out = self.bn1(out,training)
        out = self.relu1(out)

        return out


class gruBlock(tf.keras.Model):

    def __init__(self,dims=[32,32,1],input_shape=None,activation=None,name=None,final=False):
        
        super(gruBlock, self).__init__()

        self.final = final

        self.g1 = tf.keras.layers.GRU(units=dims[0], return_sequences=True, return_state=True,name=f"{name}-0",input_shape=input_shape)
        self.g2 = tf.keras.layers.GRU(units=dims[1], return_sequences=True, return_state=True,name=f"{name}-1",input_shape=input_shape)
        self.g3 = tf.keras.layers.GRU(units=dims[2], return_sequences=True, return_state=True,name=f"{name}-3",input_shape=input_shape)              

    def call(self, inp, training=False):
        
        shape = tf.shape(inp)
        x = tf.reshape(inp,[-1,shape[-2],shape[-1]])
        
        x,f = self.g1(x)
        x,f = self.g2(x)
        x,f = self.g3(x)
        
        if self.final:
            new_shape = tf.concat([shape[:-2],[-1]],0)
            out = tf.reshape(f,new_shape)

        else:
            new_shape = tf.concat([shape[:-1],[-1]],0)
            out = tf.reshape(x,new_shape)

        return out

def getSequential(dims=[32,32,1],name=None,activation=None,final=True):

    final_list = []

    for idx,n in enumerate(dims):
        if final and idx == len(dims)-1:
            final_list.append(Dense(2, activation='sigmoid',name=f"{name}-{idx}"))
        else:
            final_list.append(Dense(n, activation=activation,name=f"{name}-{idx}"))

    return tf.keras.Sequential(final_list, name=name)
