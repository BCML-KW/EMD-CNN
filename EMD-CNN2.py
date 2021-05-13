# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:45:15 2021

@author: BMCL
"""




import os
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio

import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
import random
from scipy.io import savemat
import scipy.stats as scs

tf.test.is_gpu_available()


random.seed(777)
np.random.seed(777)
fileLoc = "F:\Sleep_data\public_data\sleep-edfx_data\sleep-edfx-mat\SC_EMD_new\SC\*.mat"
filelist = glob.glob(fileLoc)

Label = np.zeros((1,1,1));
sig = np.zeros((1,3000,1));
imf = np.zeros((1,3000,6));

indices = np.random.randint(0,160949, size=500)
np.array(filelist)[indices.astype(int)]
#for loc in filelist[indices.astype(int)]:
for loc in np.array(filelist)[indices.astype(int)]:
    mat =sio.loadmat(loc)
    Label0 = mat['Label']
    sig0 = mat['sig']    
    imf0 = mat['imf']
    Label0 = Label0.reshape((1, 1, 1))
    sig0 = sig0.reshape((1,3000,1))
    imf0 = imf0.reshape((1,3000,6))
    for i in range(0,np.shape(imf0)[2]):
        imf0[:,:,i] = (imf0[:,:,i]-np.mean(imf0[:,:,i]))/np.std(imf0[0,:,i],0)
    Label = np.append(Label, Label0, axis = 0)
    sig = np.append(sig, sig0, axis = 0)
    imf = np.append(imf, imf0, axis = 0)
    
Label = np.delete(Label, 0, axis=0)    
sig = np.delete(sig, 0, axis=0)    
imf = np.delete(imf, 0, axis=0)



for j in range(0, np.shape(imf)[0]):
    for i in range(0,6):  
        imf[j,:,i] = np.mean(abs(sig[j,:,0]))/np.mean(abs(imf[j,:,i]))*imf[j,:,i]
        



training_set, test_set, training_label, test_label = train_test_split(sig, imf, train_size = 0.7, random_state=777)

num = np.zeros((np.shape(test_label)[0], np.shape(test_label)[2])) 
for j in range(0, np.shape(test_label)[0]):
    for i in range(0,6):  
        num[j,i] = np.mean(abs(test_label[j,:,i]))

#=============================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus: 
    try: # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU') 
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e: # Memory growth must be set before GPUs have been initialized
        print(e)
#======================================================================        
inputs = tfk.Input(shape=(3000,1))
x = inputs
#x= tfk.layers.Conv1D(256,3, padding='same')(x)
x= tfk.layers.Dense(1024)(x)
x= tfk.layers.BatchNormalization()(x)
x= tfk.layers.LeakyReLU(alpha=0.3)(x)


x= tfk.layers.Conv1D(512,3, padding='same')(x)
x=tfk.layers.Dropout(0.25)(x)
x= tfk.layers.LeakyReLU(alpha=0.01)(x)

x=tfk.layers.Dropout(0.25)(x)

x= tfk.layers.Conv1D(256, 3, padding='same')(x)
x=tfk.layers.Dropout(0.25)(x)
x= tfk.layers.LeakyReLU(alpha=0.3)(x)

x= tfk.layers.Conv1D(64, 3, padding='same')(x)
x= tfk.layers.BatchNormalization()(x)
x= tfk.layers.LeakyReLU(alpha=0.3)(x)

x= tfk.layers.Conv1D(32, 3, padding='same')(x)
x=tfk.layers.Dropout(0.25)(x)
x= tfk.layers.LeakyReLU(alpha=0.3)(x)


x= tfk.layers.Conv1D(6, 3, padding='same')(x)
x= Reshape((3000,6))(x)
decoder = Model(inputs, x)
decoder.summary()

inputs0 = Input(shape=(3000,1))
decoded = Lambda(lambda v: tf.cast(tf.signal.fft(tf.cast(v,dtype=tf.complex64)),tf.float32))(inputs0)
FFT = Model(inputs0, decoded)
FFT.summary()


def my_loss(y_true, y_pred):
        loss = K.mean(abs(K.sum(y_true, axis=2)))/K.mean(abs(y_true[None,:,0]))*K.mean(K.abs(y_true[None,:,0]- y_pred[None,:,0])) \
        + K.mean(abs(K.sum(y_true, axis=2)))/K.mean(abs(y_true[None,:,1]))*K.mean(K.abs(y_true[None,:,1]- y_pred[None,:,1])) \
        + K.mean(abs(K.sum(y_true, axis=2)))/K.mean(abs(y_true[None,:,2]))*K.mean(K.abs(y_true[None,:,2]- y_pred[None,:,2])) \
        + K.mean(abs(K.sum(y_true, axis=2)))/K.mean(abs(y_true[None,:,3]))*K.mean(K.abs(y_true[None,:,3]- y_pred[None,:,3])) \
        + K.mean(abs(K.sum(y_true, axis=2)))/K.mean(abs(y_true[None,:,4]))*K.mean(K.abs(y_true[None,:,4]- y_pred[None,:,4])) \
        + K.mean(abs(K.sum(y_true, axis=2)))/K.mean(abs(y_true[None,:,5]))*K.mean(K.abs(y_true[None,:,5]- y_pred[None,:,5])) \
        + K.mean(abs(K.sum(y_true, axis=2)))/K.mean(abs(K.sum(y_true, axis=2)))*K.mean(abs(K.sum(y_true,axis=2)- K.sum(y_pred, axis=2))) \
        + 0.01*K.mean(abs(FFT(K.sum(y_true, axis=2)[0:1500])))/K.mean(abs(FFT(y_true[None,:,0])[0:1500]))*K.mean(abs(abs(FFT(y_true[None,:,0])[0:1500])-abs(FFT(y_pred[None,:,0])[0:1500] ))) \
        + 0.01*K.mean(abs(FFT(K.sum(y_true, axis=2)[0:1500])))/K.mean(abs(FFT(y_true[None,:,0])[0:1500]))*K.mean(abs(abs(FFT(y_true[None,:,1])[0:1500])-abs(FFT(y_pred[None,:,1])[0:1500] ))) \
        + 0.01*K.mean(abs(FFT(K.sum(y_true, axis=2)[0:1500])))/K.mean(abs(FFT(y_true[None,:,0])[0:1500]))*K.mean(abs(abs(FFT(y_true[None,:,2])[0:1500])-abs(FFT(y_pred[None,:,2])[0:1500] ))) \
        + 0.01*K.mean(abs(FFT(K.sum(y_true, axis=2)[0:1500])))/K.mean(abs(FFT(y_true[None,:,0])[0:1500]))*K.mean(abs(abs(FFT(y_true[None,:,3])[0:1500])-abs(FFT(y_pred[None,:,3])[0:1500])))  \
        + 0.01*K.mean(abs(FFT(K.sum(y_true, axis=2)[0:1500])))/K.mean(abs(FFT(y_true[None,:,0])[0:1500]))*K.mean(abs(abs(FFT(y_true[None,:,4])[0:1500])-abs(FFT(y_pred[None,:,4])[0:1500])) )\
        + 0.01*K.mean(abs(FFT(K.sum(y_true, axis=2)[0:1500])))/K.mean(abs(FFT(y_true[None,:,0])[0:1500]))*K.mean(abs(abs(FFT(y_true[None,:,5])[0:1500])-abs(FFT(y_pred[None,:,5])[0:1500])) )
        
        return loss

decoder.compile(loss=my_loss, optimizer="adadelta", metrics=['mae'])
weights = decoder.get_weights()

early_stopping = EarlyStopping(patience = 15)
    

batchhistory=decoder.fit(training_set, training_label,
                             batch_size=4,
                             epochs=5000,
                             verbose=1,
                             validation_data=(test_set, test_label), callbacks = [early_stopping])

prediction = decoder.predict(test_set)

for j in range(0, np.shape(prediction)[0]):
    for i in range(0,6):  
        prediction[j,:,i] = num[j,i]/np.mean(abs(sig[j,:,0]))*prediction[j,:,i]


plt.subplot(421)
plt.plot(test_label[1,:,1])

plt.subplot(422)
plt.plot(prediction[1,:,1])

plt.subplot(423)
plt.plot(test_label[1,:,3])

plt.subplot(424)
plt.plot(prediction[1,:,3])

plt.subplot(425)
plt.plot(test_label[1,:,5])

plt.subplot(426)
plt.plot(prediction[1,:,5])




savemat(r'F:\EMD_GAN_project\prediction\imf_prediction0511.mat', {'prediction' : prediction })
savemat(r'F:\EMD_GAN_project\prediction\test_label0511.mat', {'test_label' : test_label })

# In MATLAB plot
#subplot(2,1,1); plot(sum(test_label(50,:,:), 3));title('original'); subplot(2,1,2);plot(sum(prediction(50, : ,:),3));title('prediction');

#===============================================
