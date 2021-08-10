# -*- coding: utf-8 -*-

"""
Created on Sun Apr 12 15:21:15 2020
This section is mainly used for feature fusion - health index generation
@author: Zihan Ren - The Pennsylvania State University - Energy and Mineral Engineering Department
"""

import numpy as np
import pandas as pd
import preprocess_PCA as pca_process
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None

def main(n_com,phm08_train,phm08_test):
    
    n_components = n_com
    
    train,test = pca_process.main(n_components,phm08_train,phm08_test)
    
    # construct PCA components
    pca_name = []
    for i in range(n_components):
        name = 'pc'+str(i+1)
        pca_name.append(name)
    
    train.loc[:,'fused_feature'] = -9999
    
    
    for i in range(6):
        # separating on first environment
        e = train[train[i]==1]
        e_test = test[test[i]==1]
        e_0 = e[e['rul']<=5]
        e_1 = e[e['rul']>=300]
        
        e_0['rul'] = 0
        e_1['rul'] = 1 
        
        
            
        # construct the features and label
        x = np.vstack((e_0[pca_name].values,e_1[pca_name].values))
        y = np.vstack((e_0['rul'].values.reshape(-1,1),e_1['rul'].values.reshape(-1,1)))
        
        reg = LinearRegression().fit(x,y)
        pca_all = e[pca_name].values
        pca_all_test = e_test[pca_name].values
        fused_feature = reg.predict(pca_all)
        fused_feature_test = reg.predict(pca_all_test)
        
        train.loc[train[i]==1,'fused_feature'] = fused_feature[:,0]
        test.loc[test[i]==1,'fused_feature'] = fused_feature_test[:,0]
        train.loc[:,'adjust'] = -train['rul'].values
        
        
    return train,test



def exp_smooth(data,parameter):
    
    data.loc[:,'smooth'] = 1
    for i in np.unique(data['machine']):
        
        mach_index = i
        m = data[data['machine']==mach_index]
        fused_feature = m['fused_feature']
        exp_smooth = fused_feature.ewm(alpha=parameter).mean()
        data.loc[data['machine']==mach_index,'smooth'] = exp_smooth
        
    return data




if __name__ == '__main__':
    
    column_name = ['machine','time','o1','o2','o3']
    sensor_name = []
    for i in range(21):
        column_name.append('s'+str(i+1))
        sensor_name.append('s'+str(i+1))
    del i
    
    phm08_train = pd.read_csv(r'kaggle\CMaps\train_FD002.txt',
                         delim_whitespace=True,header=None,names=column_name)
    phm08_test = pd.read_csv(r'kaggle\CMaps\test_FD002.txt',
                         delim_whitespace=True,header=None,names=column_name)
    
    train,test = main(7,phm08_train,phm08_test)
    
    train = exp_smooth(train,0.5)
    test = exp_smooth(test,0.5)
    # plot health index for visulization
    # train
    for i in range(5):
        sns.set()
        fig = plt.figure()
        m1 = train[train['machine'] == i+1]
        plt.scatter(m1['time'],m1['fused_feature'],label='health index')
        plt.xlabel('time/cycle')
        plt.ylabel('Health index/HI')
        plt.title('Time series health index profile in machine '+str(i+1))
        plt.legend()
        #fig.savefig('figure/hitime_m'+str(i+1))
        plt.show()
        
    # smoothed
    for i in range(5):
        sns.set()
        fig = plt.figure()
        m1 = train[train['machine'] == i+1]
        plt.scatter(m1['time'],m1['smooth'],label='smoothed health index')
        plt.xlabel('time/cycle')
        plt.ylabel('Smoothed Health index/HI')
        plt.legend()
        plt.title('Smoothed time series health index profile in machine '+str(i+1))
        fig.savefig('figure/smoo_hitime_m'+str(i+1))
        plt.show()
        
    # test
    for i in range(10):
        sns.set()
        fig = plt.figure()
        m1 = test[test['machine'] == i+1]
        plt.scatter(m1['time'],m1['smooth'],label='health index')
        plt.ylim([-0.2,1.2])
        plt.xlabel('time/cycle')
        plt.ylabel('Health index/HI')
        plt.title('Time series health index of test data profile in machine '+str(i+1))
        fig.savefig('figure/testhitime_msmoo'+str(i+1))
        plt.show()



    
    
    
    
    
    
    
    
    





