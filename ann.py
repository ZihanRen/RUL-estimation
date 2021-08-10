# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:48:24 2020
This section is mainly used for Deep Feedforward Neural Network prediction
@author: Zihan Ren - 
The Pennsylvania State University - 
Energy and Mineral Engineering Department
"""

# import PCA-process module for generate PCA components
import os
os.chdir('preprocessing')
import sample_construct as pca_sample
os.chdir('..')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.patches as mpaches 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input,Dense
import time

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
train,test = pca_sample.main(7,phm08_train,phm08_test)
label = pd.read_csv(r'preprocessing\kaggle\CMaps\RUL_FD002.txt',
                          delim_whitespace=True,header=None)

feature_space = []
for i in range(6):
    feature_space.append(i)
for i in range(7):
    feature_space.append(f'pc{i+1}')

# get the training features
X = train.loc[:,feature_space].values
# get the training label
y = train.loc[:,'rul'].values.reshape(-1,1)

scaler = MinMaxScaler()
scaler.fit(y)
y = scaler.transform(y)

start = time.perf_counter()
# deep feedforward neural network construction    
dim = X.shape[1]
input_feature = Input(shape=(dim,)) # input layer
hidd_1 = Dense(20,activation='relu')(input_feature) # 1st hidden layer
hidd_2 = Dense(30,activation='relu')(hidd_1) # 2nd hidden layer
hidd_3 = Dense(20,activation='relu')(hidd_2) # 3rd hidden layer
sigmoid = Dense(1,activation='sigmoid')(hidd_3) # 4th activation layer
predict = Dense(1,activation='linear')(sigmoid) # final activation layer

model = Model(inputs=input_feature,outputs=predict)

model.summary()
# using adam optimizer
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])
# perform 60 epochs
history = model.fit(X,y,epochs=60,validation_split=0.33)
        
# get the prediction on test unit
prediction = model.predict(test[feature_space])
prediction = pd.DataFrame(
    np.hstack((test.loc[:,['machine','time']].values,prediction))
                          ,columns = ['machine','time','prediction']
                          )
rul = []

for machine_index in np.unique(test['machine']):
    rul_m = prediction[prediction['machine']==machine_index].iloc[-1,-1]
    rul.append(rul_m)
rul = scaler.inverse_transform(np.array(rul).reshape(-1,1))

mean_absolute_error =np.mean(abs(rul - label.values.reshape(-1,1)))
mean_squared_error = np.mean(np.square(rul - label.values.reshape(-1,1)))


#np.savetxt('SBM_result_ann.txt',rul,fmt='%i')

# err_array.append(error)
# mse.append(mean_absolute_error)
# mae.append(mean_squared_error)


finish = time.perf_counter()

print(f'Finished in {round(finish-start,2)} seconds(s)')










######################### visulization ##########################

def arr_rul(rul,t):
        rul_arr = []
        for i in range(len(t)):
            rul_arr.append(rul+i)
            
        rul_arr.reverse()
        return rul_arr
    # generate test matrix
    
for i in range(10):
    machine_index = i+1
    m1 = test.loc[test['machine'] == machine_index,:]
    t = m1['time']
    rul = label.iloc[machine_index-1,:].values[0]
    
    label_rul = arr_rul(rul,t)
    
    # get the prediction array
    model_pred = scaler.inverse_transform(model.predict(m1[feature_space]))
    
    # visulization of degradation pattern
    sns.set()
    fig = plt.figure()
    plt.plot(t,label_rul,c='r',label='ground truth')
    plt.scatter(t,model_pred,label = 'prediction') 
    plt.xlabel('time')
    plt.ylabel('RUL')
    plt.title(f'Prediction of Life profile in machine {machine_index}')
    # fig.savefig(f'figure/annpredm{machine_index}')
    

# validation visulization of label vs prediction on multiple machines
sample_num = 100
rul = []
for machine_index in np.unique(test['machine'])[0:sample_num]:
        rul_m = prediction[prediction['machine']==machine_index].iloc[-1,-1]
        rul.append(rul_m)
rul = scaler.inverse_transform(np.array(rul).reshape(-1,1))


#  Visulization of overall performance
sns.set()
fig = plt.figure()
plt.plot(np.unique(test['machine'])[0:sample_num],label.iloc[0:sample_num,:],
         c='r',label='ground truth')
plt.plot(np.unique(test['machine'])[0:sample_num],rul,
         c='b',label='Prediction using ANN')
plt.xlabel('Machines Index')
plt.ylabel('Remaining Useful Cycles')
plt.legend()
#fig.savefig(f'figure/ann10machines.png')
    






















































