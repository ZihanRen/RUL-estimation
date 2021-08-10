# -*- coding: utf-8 -*-

"""
Created on Mon May  4 22:44:59 2020
Created for model library and clusters
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
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D





def main(n_cluster,phm08_train,phm08_test):


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
        # calculate the standard devation of this regression model
        mspe = np.mean( np.sqrt(np.square(y_hat - y)) )
        return mspe,popt
    
    
    def model_cluster(model_sequence,n_group):
      # model cluster
      pattern_group =  \
      np.array(model_sequence).reshape(
          len(model_sequence),len(model_sequence[0])
          )
      
      from sklearn.cluster import KMeans
           
      n_group = n_group
      kmean = KMeans(n_clusters=n_group,random_state=10)
      cluster_label = kmean.fit_predict(pattern_group)
      # find cluster representative, center of cluster
      rep_seq = kmean.cluster_centers_
      
      return cluster_label,rep_seq
      
    
    train,test = pca_sample.main(7,phm08_train,phm08_test)
    train = pca_sample.exp_smooth(train,0.5)
    test = pca_sample.exp_smooth(test,0.5)
    cycle = cycle_generate(400)
    
    model_sequence = []
    model_par = []
    lib = []
    uncertainty = []
    
    
    for i in np.unique(train['machine']):
        
        m = train[train['machine']==i]
        # output the model parameters and 
        # standard deviation(mean square prediction error)
        mspe,popt = model_output(exp_func,m['adjust'].values,
                                 m['smooth'].values)        
        
        sequence = exp_func(cycle,*popt)
        model_sequence.append(sequence)
        matrix_seq = np.hstack((
            np.hstack((sequence,cycle)),
            np.hstack((sequence + mspe,sequence - mspe)) 
            ))
        
        model_lib = pd.DataFrame(
            matrix_seq,columns=['sequence','cycle','top','bottom']
            )
        
        lib.append(model_lib)
        model_par.append(popt)
        uncertainty.append(mspe)
        
    cluster_label,rep_seq = model_cluster(model_sequence,n_cluster)
    
    return cluster_label,rep_seq,lib,train,test    




if __name__ == '__main__':
    # some data preparation
    column_name = ['machine','time','o1','o2','o3']
    sensor_name = []
    for i in range(21):
        column_name.append('s'+str(i+1))
        sensor_name.append('s'+str(i+1))
    del i
    
    phm08_train = pd.read_csv(r'preprocessing\kaggle\CMaps\train_FD002.txt',
                         delim_whitespace=True,header=None,names=column_name)
    phm08_test = pd.read_csv(
        r'preprocessing\kaggle\CMaps\test_FD002.txt',
        delim_whitespace=True,header=None,names=column_name
                             )
        
            
    cluster_label,rep_seq,lib,train,test = main(10,phm08_train,phm08_test)






    # visulize 10 clusters in sequence plot using 100 smaples
    num_samples = 100
    color_list = ['aqua','aquamarine','fuchsia','black','blue','brown','coral','gold','green','lime']
    cluster_label  = cluster_label[0:num_samples]

    custom_lines = [Line2D([0], [0], color = color_list[0], lw=1),
                Line2D([0], [0], color = color_list[9], lw=1),
                Line2D([0], [0], color = color_list[1], lw=1),
                Line2D([0], [0], color = color_list[2], lw=1),
                Line2D([0], [0], color = color_list[3], lw=1),
                Line2D([0], [0], color = color_list[4], lw=1),
                Line2D([0], [0], color = color_list[5], lw=1),
                Line2D([0], [0], color = color_list[6], lw=1),
                Line2D([0], [0], color = color_list[7], lw=1),
                Line2D([0], [0], color = color_list[8], lw=1),]
    
    


    fig,ax = plt.subplots()
    for cluster in range(10):
        
        for j in list(np.where(cluster_label==cluster))[0]:
            
            plt.plot(lib[j]['cycle'],lib[j]['sequence'],color=color_list[cluster])
            
    
    plt.xlim([-500,0])
    plt.xlabel('cycle')
    plt.ylabel('health index')
    ax.legend(custom_lines,['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10'])
    

    
    
    
    # Visualize the representative sequence
    for i in range(10):
        plt.plot(lib[0]['cycle'],rep_seq[i],label=f'r{i+1}')
    plt.xlim([-500,0])
    plt.legend()
    plt.xlabel('cycle')
    plt.ylabel('health index')
    
    
        
        
        
        
        
        
        
        
        
        
        
    # visulization
    for i in range(10):
        fig = plt.figure()
        m = train[train['machine'] == i+1]
        plt.plot(lib[i]['cycle'],lib[i]['sequence'],'r')
        plt.plot(lib[i]['cycle'],lib[i]['top'],'r--',label='Reference model boundary')
        plt.plot(lib[i]['cycle'],lib[i]['bottom'],'r--')
        plt.fill_between(lib[i]['cycle'], lib[i]['top'],lib[i]['bottom'],alpha = 0.2)
        plt.scatter(m['adjust'],m['smooth'],label='training health index')
        plt.title('Model'+str(i+1)+' vs fused feature')
        plt.xlabel('adjust cycle')
        plt.ylabel('scaled RUL')
        plt.legend()
        # fig.savefig(f'figure/trainboundary_m{i+1}')
        plt.show()







