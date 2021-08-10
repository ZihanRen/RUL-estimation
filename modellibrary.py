# -*- coding: utf-8 -*-

"""

Created on Mon Apr 13 20:45:37 2020
This part is mainly for building up model library in the similarity based method
@author: Zihan Ren - The Pennsylvania State University - Energy and Mineral Engineering Department
"""

# some python modulues should be uploaded through subfolder
import os
os.chdir('preprocessing')
import sample_construct as pca_sample
os.chdir('..')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns


def model_cluster(model_sequence):
      # model cluster
      pattern_group = np.array(model_sequence).reshape(len(model_sequence),len(model_sequence[0]))
      
      from sklearn.cluster import KMeans
      from sklearn.metrics import silhouette_score
      
       
      n_group = 2
      kmean = KMeans(n_clusters=n_group,random_state=10)
      cluster_label = kmean.fit_predict(pattern_group)
      
      return cluster_label,pattern_group 


def main(n_com,phm08_train,phm08_test):
    
    
    def cycle_generate(max_num):
        '''
        The max_num should be the t_max, time limit in the model
        '''
        time_limit = np.arange(1,max_num+2)
        time_limit = np.flip(time_limit).reshape(-1,1)
        cycle = 1 - time_limit
        return cycle.reshape(-1,1)
    
    def exp_func(x,a,b):
        # exponential regression
        return a*(np.exp(b*x)-1)
    
    
    def model_output(function,x,y):
        '''
        The input x and y should be reshape as (-1,1)
        '''
        popt,pcov = curve_fit(function,x,y)
        y_hat = function(x,*popt)
        mspe = np.mean(np.square(y_hat - y))
        return mspe,popt
    
    
    
    train,test = pca_sample.main(n_com,phm08_train,phm08_test)
    train = pca_sample.exp_smooth(train,0.5)
    test = pca_sample.exp_smooth(test,0.5)
    cycle = cycle_generate(400)
    
    model_sequence = []
    model_par = []
    lib = []
    uncertainty = []
    
    
    for i in np.unique(train['machine']):
        
        m = train[train['machine']==i]
        # output the model parameters and standard deviation(mean square prediction error)
        mspe,popt = model_output(exp_func,m['adjust'].values,
                                 m['smooth'].values)        
        
        sequence = exp_func(cycle,*popt)
        model_sequence.append(sequence)
        model_lib = pd.DataFrame(np.hstack((sequence,cycle)),columns=['sequence','cycle'])
        lib.append(model_lib)
        model_par.append(popt)
        uncertainty.append(mspe)
    
    return model_par,lib,test,model_sequence,train,uncertainty






# visulization
if __name__ == '__main__':
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
    
    
    
    model_par,lib,test,model_sequence,train,uncertainty = main(7,phm08_train,phm08_test)
    cluster_label,pattern_group = model_cluster(model_sequence)
    
    
    cycle= lib[0]['cycle'].values
    
    
    
    
    # cluster based on parameters
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
      
    
    
    
    # clustering on model parameters
    par_arr = np.array(model_par)
    score_arr = []
    # compute the siloutte score to find the optimal clusters
    for i in range(2,17):
        
        n_group = i
        kmean = KMeans(n_clusters=n_group,random_state=10)
        cluster_label = kmean.fit_predict(par_arr)
        score = silhouette_score(par_arr,cluster_label)
        plt.scatter(n_group,score,c='r')
        score_arr.append(score)
    plt.ylim([0,1])
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Score Changing Profile Regarding With Number Of Clusters')
    plt.show()
      
    print('The optimal group number is 2')

    n_group = 10
    kmean = KMeans(n_clusters=n_group,random_state=10)
    cluster_label = kmean.fit_predict(par_arr)
    
    # visulization
    for i in range(n_group):
        m = par_arr[np.where(cluster_label==i)]
        color = (i+1)/10
        plt.scatter(m[:,0],m[:,1],cmap=color)
        plt.xlim([-1.2,-0.6])
        plt.ylim([0,0.04])
    
    
    
    pattern_group = np.array(model_sequence).reshape(len(model_sequence),len(model_sequence[0]))
    
    
    
    
    
    
    
    
    
    
    
    ###################### for visulization ##############################
    ### plot the visulization of different model group ####
    from random import random
    fig = plt.figure()
    sns.set()

    for i in range(n_group):
        
        modeli = pattern_group[np.where(cluster_label==i)]
        
        
        color = (random(),random(),random())
        
        
        # red_patch = mpaches.Patch(color='red',label='model1')
        # blue_patch = mpaches.Patch(color='blue',label='model2')
        
        # visulize the 200 models cluster
        for j in range(8):
            plt.plot(cycle,modeli[j,:],c=color)

        plt.xlabel('cycle')
        plt.ylabel('health index')
        plt.title('models cluster visulization')
        # plt.legend(handles=[red_patch,blue_patch])
    # fig.savefig('figure/model_cluster100.png')
    
    
    
    
    # simply plot multiple models
    fig = plt.figure()
    for i in range(259):
        plt.plot(lib[i]['cycle'],lib[i]['sequence'],c='b')
    plt.xlabel('Adjusted cylce')
    plt.ylabel('Health index')
    plt.title('259 Models health index profile')
        
    
    
    
    
    
    
    
    # multiple models with health index on the same figure
    fig = plt.figure()
    for i in range(5):
        m = train[train['machine'] == i+1]
        plt.plot(lib[i]['cycle'],lib[i]['sequence'])
        plt.scatter(m['adjust'],m['smooth'])
    plt.xlabel('adjust cycle')
    plt.ylabel('scaled RUL')
    plt.title('Health index fitted by several models')
    # fig.savefig('preprocessing/figure/modelvsHI.png')
    
    
    
    # plot multiple figures with different models and fused feature - single
    for i in range(4):
        fig = plt.figure()
        m = train[train['machine'] == i+1]
        plt.plot(lib[i]['cycle'],lib[i]['sequence'],'r',label='Model')
        plt.scatter(m['adjust'],m['smooth'],label='smoothed health index')
        plt.xlabel('adjust cycle')
        plt.ylabel('scaled RUL')
        plt.legend()
        plt.title('Model'+str(i+1)+' vs fused feature')
        plt.show()
        #fig.savefig('preprocessing/figure/model'+str(i+1)+'_fusedfeature.png')
    
    
    
    # plot the shifted health index versus model
    
    fig = plt.figure()
    m = train[train['machine'] == 1]


    for i in range(6):
        cycle_shift = (i)*20
    
        plt.plot(lib[0]['cycle'],lib[0]['sequence'],'r',label='Model')

        plt.scatter(m['adjust']-cycle_shift,m['smooth'])
    plt.xlabel('adjust cycle')
    plt.ylabel('scaled RUL')
    plt.title('Shifted sequence ')
    
    
    
    plt.plot(lib[0]['cycle'],lib[0]['sequence'],'r',label='Model')
    
    plt.scatter(m['adjust']-100,m['smooth'])

    plt.scatter(m['adjust'],m['smooth'])
    plt.xlabel('adjust cycle')
    plt.ylabel('scaled RUL')
    
    
    
    # parameter distribution
    par = np.array(model_par)  
    fig = plt.figure()
    plt.scatter(par[:,0],par[:,1])
    plt.xlabel('parameter a')
    plt.ylabel('parameter b')
    plt.title('Parameter distribution in model library')
    # fig.savefig('preprocessing/figure/paradist1.png')
    
    
    
    
    
    
    
    
    
    
    