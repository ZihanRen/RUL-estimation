# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:49:21 2020
This section is mainly used for Deep recurrent Neural Network prediction
@author: Zihan Ren - 
The Pennsylvania State University - 
Energy and Mineral Engineering Department
"""

import os
os.chdir('preprocessing')
import sample_construct as pca_sample
os.chdir('..')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpaches 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
import time

from numpy.random import seed
seed(1)

def model_history(layer_num,time_step,dim,feature,label,epoch):
    
    '''
    Construct the recurrent layer: 2 and 3
    '''
    
    
    if layer_num == 3:
        # fit training sequence into simple RNN
        feature_input = Input(shape = (time_step,dim))
        rnn1 = SimpleRNN(20,return_sequences=True, activation='relu')(feature_input)
        rnn2 = SimpleRNN(15,return_sequences=False,activation='relu')(rnn1)
        output = Dense(1,activation='sigmoid')(rnn2)
        model = Model(inputs=feature_input,outputs=output)
        model.summary()
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error'])
        history = model.fit(
            x=feature,y=label,
            epochs = epoch,
            validation_split=0.33
            )
    
        
    elif layer_num==2:
        feature_input = Input(shape = (time_step,dim))
        
        rnn1 = SimpleRNN(30)(feature_input)
        sigmoid = Dense(1,activation='sigmoid')(rnn1)
        output = Dense (1,activation='linear')(sigmoid)
        model = Model(inputs=feature_input,outputs=output)
        model.summary()
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error'])
        history = model.fit(x=feature,y=label,epochs = epoch)
        
        
    elif layer_num == 'deep':
        feature_input = Input(shape = (time_step,dim))
        rnn1 = SimpleRNN(30,return_sequences=True)(feature_input)
        rnn2 = SimpleRNN(10)(rnn1)

        dense1 = Dense(30,activation='relu')(rnn2)
        dense2 = Dense(10,activation='relu')(dense1)
        sigmoid = Dense(1,activation='sigmoid')(dense2)
        output = Dense (1,activation='linear')(sigmoid)
        model = Model(inputs=feature_input,outputs=output)
        model.summary()
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error'])
        history = model.fit(x=feature,y=label,epochs = epoch)
        
        
    elif layer_num == 'LSTM':
        feature_input = Input(shape = (time_step,dim))
        
        rnn1 = LSTM(30)(feature_input)
        sigmoid = Dense(1,activation='sigmoid')(rnn1)
        output = Dense (1,activation='linear')(sigmoid)
        model = Model(inputs=feature_input,outputs=output)
        model.summary()
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error'])
        history = model.fit(x=feature,y=label,epochs = epoch)
        
    return history


def rnnlayer(layer_num,time_step,dim,feature,label,epoch):
    
    '''
    The different option simply implied that different recurrent architecture 
    for validation purpose
    '''
    
    
    if layer_num == 3:
        # staked RNN with one dense layer
        # fit training sequence into simple RNN
        feature_input = Input(shape = (time_step,dim))
        rnn1 = SimpleRNN(20,return_sequences=True)(feature_input)
        rnn2 = SimpleRNN(15,return_sequences=False)(rnn1)
        output = Dense(1,activation='sigmoid')(rnn2)
        model = Model(inputs=feature_input,outputs=output)
        model.summary()
        model.compile(optimizer='adam',loss='mean_squared_error',
                      metrics=['mean_absolute_error'])
        model.fit(x=feature,y=label,epochs = epoch,validation_split=0.33)
    
        
    elif layer_num==2:
        # simple RNN with one recursive layer
        feature_input = Input(shape = (time_step,dim))
        
        rnn1 = SimpleRNN(30)(feature_input)
        sigmoid = Dense(1,activation='sigmoid')(rnn1)
        output = Dense (1,activation='linear')(sigmoid)
        model = Model(inputs=feature_input,outputs=output)
        model.summary()
        model.compile(optimizer='adam',loss='mean_squared_error',
                      metrics=['mean_absolute_error'])
        model.fit(x=feature,y=label,epochs = epoch)
        
        
    elif layer_num == 'deep':
        # deep stacked RNN with dense layer and two recursive layers
        feature_input = Input(shape = (time_step,dim))
        rnn1 = SimpleRNN(30,return_sequences=True)(feature_input)
        rnn2 = SimpleRNN(10)(rnn1)

        dense1 = Dense(30,activation='relu')(rnn2)
        dense2 = Dense(10,activation='relu')(dense1)
        sigmoid = Dense(1,activation='sigmoid')(dense2)
        output = Dense (1,activation='linear')(sigmoid)
        model = Model(inputs=feature_input,outputs=output)
        model.summary()
        model.compile(optimizer='adam',loss='mean_squared_error',
                      metrics=['mean_absolute_error'])
        model.fit(x=feature,y=label,epochs = epoch)
        
        
    elif layer_num == 'LSTM':
        # try long short term memory
        feature_input = Input(shape = (time_step,dim))
        
        rnn1 = LSTM(30)(feature_input)
        sigmoid = Dense(1,activation='sigmoid')(rnn1)
        output = Dense (1,activation='linear')(sigmoid)
        model = Model(inputs=feature_input,outputs=output)
        model.summary()
        model.compile(optimizer='adam',loss='mean_squared_error',
                      metrics=['mean_absolute_error'])
        model.fit(x=feature,y=label,epochs = epoch)
        
    return model

def timeseriesgenerate(train,feature_space,time_step):
    # generate sequence for training and testing
    feature = []
    label = []
    
    for i in np.unique(train.loc[:,'machine'].values):
    
        selection_data = train[train['machine']==i]
        select_feature = selection_data[feature_space].values
        select_label = selection_data['rul'].values
    
        for j in range(time_step,len(selection_data)):
            
            feature.append(select_feature[j-time_step:j,:])
            label.append(select_label[j])
    
    
    feature,label = np.array(feature),np.array(label)
    return feature,label


# data preparation
column_name = ['machine','time','o1','o2','o3']
sensor_name = []

for i in range(21):
    column_name.append('s'+str(i+1))
    sensor_name.append('s'+str(i+1))
del i


phm08_train = pd.read_csv(r'preprocessing\kaggle\CMaps\train_FD002.txt',
                     delim_whitespace=True,header=None,names=column_name)
phm08_test = pd.read_csv(r'preprocessing\kaggle\CMaps\test_FD002.txt',
                     delim_whitespace=True,header=None,names=column_name)
train,test = pca_sample.main(9,phm08_train,phm08_test)
test_label = pd.read_csv(r'preprocessing\kaggle\CMaps\RUL_FD002.txt',
                          delim_whitespace=True,header=None)

feature_space = []
for i in range(6):
    feature_space.append(i)
for i in range(9):
    feature_space.append(f'pc{i+1}')

dim = len(feature_space)

# scale the training label
scaler = MinMaxScaler()
scaler.fit(np.unique(train['rul'].values).reshape(-1,1))

# prepare training sequences and paramter tuning:

time_step = 13 # really close to longest sequence in testing machine
layer_num = 3
epoch = 7


# generate time series features and labels
feature,label = timeseriesgenerate(train,feature_space,time_step)
label = scaler.transform(label.reshape(-1,1))
start = time.perf_counter()

# fit training sequence into simple RNN
model = rnnlayer(layer_num,time_step,
                  len(feature_space),feature,label,epoch)
prediction = []
        
for machine_index in np.unique(test['machine']):
    # test prediction
    test_feature = []
    selection_data = test[test['machine']==machine_index]
    select_feature = selection_data[feature_space].values
    
    for i in range(time_step,len(selection_data)):
        test_feature.append(select_feature[i-time_step:i,:])
    test_feature = np.array(test_feature)
    prediction.append(model.predict(test_feature)[-1,0])
    
prediction = scaler.inverse_transform(np.array(prediction).reshape(-1,1))
error = np.sum(abs(test_label.values.reshape(-1,1) - prediction))
mean_absolute_error =np.mean(
    abs(prediction  - test_label.values.reshape(-1,1))
    )
mean_squared_error = np.mean(
    np.square(prediction - test_label.values.reshape(-1,1))
    )
#np.savetxt('SBM_result_rnn.txt',prediction,fmt='%i')

finish = time.perf_counter()
print(f'Finished in {round(finish-start,2)} seconds(s)')



######################## visulization ###########################

def arr_rul(rul,t):
    rul_arr = []
    for i in range(len(t)):
        rul_arr.append(rul+i)
        
    rul_arr.reverse()
    return rul_arr


def time_series_generate(feature_vec,time_step):
    output_feature = []
    for i in range(time_step,len(feature_vec)):
        output_feature.append(feature_vec[i-time_step:i,:])
    return np.array(output_feature)

# Visulization on 10 samples each individually
for i in range(10):
    machine_index = i+1
    m = test[test['machine']==machine_index].reset_index()
    t = m['time']
    rul_m = test_label.iloc[machine_index-1,:].values[0]
    
    label_rul = arr_rul(rul_m,t)
    
    
    t_pred = m.loc[time_step:,'time']
    test_feature = time_series_generate(m[feature_space].values, time_step)
    rul = scaler.inverse_transform(model.predict(test_feature))
    
    sns.set()
    fig = plt.figure()
    plt.plot(t,label_rul,c='r',label='ground truth')
    plt.scatter(t_pred,rul,label='prediction')
    plt.xlabel('time')
    plt.ylabel('RUL')
    plt.title(f'Prediction of Life profile in machine {machine_index} using RNN')
    # fig.savefig(f'figure/rnnpredm{machine_index}')


# validation visulization of label vs prediction on multiple machines
sample_num = 100
rul = []

for machine_index in np.unique(test['machine'])[0:sample_num]:
    # test prediction
    test_feature = []
    selection_data = test[test['machine']==machine_index]
    select_feature = selection_data[feature_space].values
    
    for i in range(time_step,len(selection_data)):
        test_feature.append(select_feature[i-time_step:i,:])
    test_feature = np.array(test_feature)
    rul.append(model.predict(test_feature)[-1,0])
    
    
rul = scaler.inverse_transform(np.array(rul).reshape(-1,1))


#  Visulization of overall performance
sns.set()
fig = plt.figure()
plt.plot(np.unique(test['machine'])[0:sample_num],test_label.iloc[0:sample_num,:],
         c='r',label='ground truth')
plt.plot(np.unique(test['machine'])[0:sample_num],rul,
         c='b',label='Prediction using RNN')
plt.xlabel('Machines Index')
plt.ylabel('Remaining useful cycles')
plt.legend()
# fig.savefig(f'figure/rnn10.png')














