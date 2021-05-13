# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 12:06:18 2021

@author: BMCL
"""

import os
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
import pandas_datareader as pdr
import scipy.io as sio
from datetime import datetime
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
import random
from scipy.io import savemat
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Reshape, Flatten, Conv1DTranspose
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import time
# Keras 가 Tensorflow 를 벡엔드로 사용할 수 있도록 설정합니다.
os.environ["KERAS_BACKEND"] = "tensorflow"

# 실험을 재현하고 동일한 결과를 얻을 수 있는지 확인하기 위해 seed 를 설정합니다.
np.random.seed(10)

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

# Generator 만들기
# 이 메서드는 크로스 엔트로피 손실함수 (cross entropy loss)를 계산하기 위해 헬퍼 (helper) 함수를 반환합니다.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mean_abs_err = tf.keras.losses.MeanAbsoluteError()

def get_generator():
    generator = Sequential()
    generator.add(Input(shape=(3000,1)))
    generator.add(Conv1D(16, 2, padding='same'))
    generator.add(LeakyReLU(0.2))

    
    generator.add(Conv1D(64, 2, padding='same'))
    generator.add(LeakyReLU(0.2))

    
    generator.add(Conv1D(256, 2, padding='same'))
    generator.add(LeakyReLU(0.2))


    generator.add(Conv1D(256,2, padding='same'))
    generator.add(LeakyReLU(0.2))

    
    generator.add(Conv1D(64,2, padding='same'))
    generator.add(LeakyReLU(0.2))
    generator.add(Dropout(0.5))
        
    generator.add(Conv1D(16,2, padding='same'))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv1D(10,2, padding='same'))
    generator.summary()
    #generator.compile(loss='binary_crossentropy', optimizer='adam')
    return generator

def generator_loss(fake_output, real_output):
    #return mean_abs_err(real_output, fake_output)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#generated_signal = generator.predict(sig[1,:,0].reshape((1, 3000,1)))
#plt.plot(generated_signal[0,3000,0,3])

# Discriminator 만들기
def get_discriminator():
    discriminator = Sequential()
    discriminator.add(Input(shape=(3000,10)))
   # discriminator.add(Reshape(target_shape=(3000,10)))
    discriminator.add(Conv1D(32,2, padding='same', activation='relu'))
    discriminator.add(Dropout(0.5))
    discriminator.add(Conv1D(128,2, padding='same', activation='relu'))
    discriminator.add(Dropout(0.5))
    discriminator.add(Conv1D(512,2, padding='same', activation='relu'))
    discriminator.add(Dropout(0.5))
    discriminator.add(Conv1D(1,1, padding='same', activation='relu'))
    discriminator.add(Dropout(0.5))
    discriminator.add(Flatten())
    discriminator.add(Dropout(0.5))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.summary()
    #discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#decision = discriminator.predict(generated_signal)

   

# `tf.function`이 어떻게 사용되는지 주목해 주세요.
# 이 데코레이터는 함수를 "컴파일"합니다.
@tf.function
def train_step1(signal, real_imf):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      fake_imf = generator(signal, training=False)

      real_output = discriminator(real_imf, training=True)
      fake_output = discriminator(fake_imf, training=True)

      gen_loss = generator_loss(fake_output, real_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    #gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    #generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train_step2(signal, real_imf):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      fake_imf = generator(signal, training=True)

      real_output = discriminator(real_imf, training=False)
      fake_output = discriminator(fake_imf, training=False)

      gen_loss = generator_loss(fake_output, real_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    #gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    #discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(training_set, training_label, epochs, BATCH_SIZE):
  for epoch in range(epochs):
    start = time.time()

    batch_count = np.shape(training_set)[0] // BATCH_SIZE
    for ii in range(batch_count):
            signal = training_set[ii*BATCH_SIZE:(ii+1)*BATCH_SIZE, :,:]
            real_imf = training_label[ii*BATCH_SIZE:(ii+1)*BATCH_SIZE, :,:]
            train_step1(signal, real_imf)
            train_step2(signal, real_imf)
    if (epoch + 1) % 5 == 0:
        fake_imf = generator.predict(signal)

        decision0=discriminator.predict(fake_imf)
        print(decision0)
        fake_imf = generator(signal, training=False)
    
        real_output = discriminator(real_imf, training=False)
        fake_output = discriminator(fake_imf, training=False)
    
        gen_loss = generator_loss(fake_output, real_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        print(r'gen_loss={}'.format(gen_loss))
        print(r'disc_loss={}'.format(disc_loss))
        if (np.mean(decision0)>0.99) & (disc_loss<0.01):
            break

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

 
            
            
     
if __name__ == '__main__':
    fileLoc = "F:\Sleep_data\public_data\sleep-edfx_data\sleep-edfx-mat\SC_BEMD\*.mat"
    filelist = glob.glob(fileLoc)
    
    Label = np.zeros((1,1,1));
    sig = np.zeros((1,3000,1));
    imf = np.zeros((1,3000,10));
    
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
        imf0 = imf0.T.reshape((1,3000,10))
        
        Label = np.append(Label, Label0, axis = 0)
        sig = np.append(sig, sig0, axis = 0)
        imf = np.append(imf, imf0, axis = 0)
        
    Label = np.delete(Label, 0, axis=0)    
    sig = np.delete(sig, 0, axis=0)    
    imf = np.delete(imf, 0, axis=0)        
        
    training_set, test_set, training_label, test_label = train_test_split(sig, imf, train_size = 0.7, random_state=777)
    
    generator = get_generator()
    discriminator = get_discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    train(training_set, training_label, 2000,4)


    
    
# =============================================================================
    signal = training_set[0:4,:,:]
    real_imf = training_label[0:4,:,:]
    fake_imf = generator.predict(signal)
    plt.subplot(421)
    plt.plot(real_imf[0,:,1])
    
    plt.subplot(422)
    plt.plot(fake_imf[0,:,1])
    
    plt.subplot(423)
    plt.plot(real_imf[0,:,4])
    
    plt.subplot(424)
    plt.plot(fake_imf[0,:,4])

    plt.subplot(425)
    plt.plot(real_imf[0,:,7])
    
    plt.subplot(426)
    plt.plot(fake_imf[0,:,7])
    
    plt.subplot(427)
    plt.plot(real_imf[0,:,9])
    
    plt.subplot(428) 
    plt.plot(fake_imf[0,:,9])
    
    
    
    
    
