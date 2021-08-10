# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:51:35 2020
This section is focused on prediction using 
original similarity based method

@author: Zihan Ren - 
The Pennsylvania State University - 
Energy and Mineral Engineering Department
"""

# import modellibrary module
import modellibrary as mlib
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn as sns

    
def main(machine_index):
    
    '''
    This part of main function is to make the 
    prediction of one test unit given test machine index
    '''
    
    
    def range_tau(model_length,m1):
    
        # define the parameter of model time attribute
        r = len(m1) # number of cycles in test sequence
        tau_min = 0 # minimum of remaining cycles 
        tau_max = model_length - r # maximum remaining cycles
        tau_range = np.arange(tau_min,tau_max+1)
    
        return tau_range
    

    def sequence_output(s1,location,r):
    
        '''
        This function is mainly used for outputing 
        the model sequence for comparing with test sequence
        s1: selected model sequence
        location: 
        remaining cycles indicating the location of model sequence
        m1: the output sequence length
        '''
        # get the beginning and end time in cycle
        cycle_begin = -location - r+1
        cycle_end = -location -r + r
        
        # construct the comparing sequence in model
        s_temp = s1[
            (s1['cycle'] >= cycle_begin) & (s1['cycle'] <= cycle_end)
            ]
        
        return s_temp.loc[:,'sequence'].values
        
    
    def sim_score(vec1,vec2):
        
        '''
        P(M|S) = P(S|M)*P(M)
        '''
        
        diff = vec1.reshape(-1,1) - vec2.reshape(-1,1)
        return np.sum(diff**2)
    
    
    def score_sort(model_result,lib):
        
        # get the most similar models and rank them
        model_stack = np.hstack(( 
            model_result,
            np.arange(0,len(lib)).reshape(-1,1) 
            ))
        model_sort = model_stack[model_stack[:,0].argsort()]
        
        return model_sort
    
    
    def weight_predict(model_sort,k,sequence_len):
        
        '''
        get the k nearest prediction
        k should be even number
        w = (0.5 + 0.5^2 + 0.5^3 + ... + 0.5^k-1) + 0.5^k-1
        '''
        
        # filter the unexpected outliers
        # sequence length should not be larger than 310 or lower than
        # 120 in most situations. If few samples left, then the result will 
        # be the averaged prediction
        
        select_matrix = sequence_len + model_sort[0:k,1]
        select_matrix = select_matrix[
            (select_matrix<310)&(select_matrix>120) 
            ]
        
        if len(select_matrix) < 5:
            prediction_matrix = model_sort[0:k,1]
            
        else:
        
            prediction_matrix = select_matrix - sequence_len
        
        return np.mean(prediction_matrix)
    
    # make a prediction on one unit
    m1 = test[test['machine'] == machine_index]
    # construct the matrix of output 
    # the first column has attribute of score 
    # and second column has attribute of tau - remaining useful life
    model_result = np.zeros((len(lib),2))
    tau_array = range_tau(len(lib[0]),m1)
    
    for model in range(len(lib)):
        # looping models in model library
        s1 = lib[model]
        # get the collection of score given different tau
        score = []
        for location in tau_array:
            sequence_model = sequence_output(s1,location,len(m1))
            distance = sim_score(sequence_model,m1.loc[:,'smooth'].values)
            score.append(distance)    
        
        
        # find the best score with lowest value and corresponding tau
        best_score = min(score)
        best_tau = tau_array[score.index(min(score))]
        
        # get the result of current machine similarity condition 
        # - tau and corresponding score
        model_result[model,0] = best_score
        model_result[model,1] = best_tau
    
    # sort the model and output sorted result and sorted score
    model_sort = score_sort(model_result,lib)
    rul = weight_predict(model_sort,20,len(m1))
    # get the prediction
    return rul,model_sort
    


# begin the prediction
if __name__ == '__main__':
     # load data
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
    
    label = pd.read_csv(r'preprocessing\kaggle\CMaps\RUL_FD002.txt',
                          delim_whitespace=True,header=None)
    # get the model library
    model_par,lib,test,model_sequence,train,uncertainty \
    = mlib.main(9,phm08_train,phm08_test)
    
    del model_sequence,phm08_test,
    phm08_train,model_par,sensor_name,column_name
    
    
    machine_num = np.unique(test['machine']).tolist()
    # number of testing samples is 259 
    sample_num = 259
    machine_num = machine_num[0:sample_num]

    # begin prediction
    start = time.perf_counter()
    rul_arr = []
    sort_list = []
    
    for machine_index in machine_num:
        print(f'   Machine {machine_index}:')
        print('...prediction begins...')
        rul,model_sort = main(machine_index)
        rul_arr.append(rul)
        sort_list.append(model_sort)
        print('...prediction complete...\n')
    
    finish = time.perf_counter()
    
    

    
    mean_absolute_error = np.mean(abs(label.iloc[0:sample_num].values.reshape(-1,1) - 
                          np.array(rul_arr).reshape(-1,1)))
    
    mean_squared_error = np.mean((label.iloc[0:sample_num].values.reshape(-1,1) - 
                          np.array(rul_arr).reshape(-1,1))**2)
    
    
    # output the prediction results for further visualizations
    # np.savetxt('SBM_result.txt',np.array(rul_arr),fmt='%i')
    
    
    
    
    
    ########################## Result Visualization Process ##########################################
    ### the main purpose of this part is to investigate relationship between 
    # reference models and test health index relationship
    
    
    ### 1st figure ###
    # Plot the top k most similar reference models degradation profile versus health index in several testing asset
    k=20
    for machine_index in range(sample_num):
        m = test[test['machine']==machine_index+1]['fused_feature']
        pred = np.round(rul_arr[machine_index])
        t = np.arange(-pred-len(m)+1,-pred+1)
        
        fig = plt.figure()
        for model_index in sort_list[machine_index][0:k,2]:
            plt.plot(lib[int(model_index)]['cycle'],lib[int(model_index)]['sequence'],c='r')
        plt.scatter(t,m,label='testing asset health index')
        plt.xlabel('Adjusted Cycle')
        plt.ylabel('Health Index')
        plt.legend()
        # fig.savefig(f'figure/m{machine_index+1}HItop20.png')
    
    
    
    ### 2nd figure ###
    ## plot the most similar reference model versus health index profile in testing asset. This part of plot is mainly 
    ## for tutorial purpose, demonstrating how \tau can be obtained through shifting the sequence
    
    # get the health index of a testing machine 3  
    machine_index = 2
    m = test[test['machine']==machine_index+1]['fused_feature']
    pred = np.round(rul_arr[machine_index])
    # t corresponds to optimal value of \tau
    t = np.arange(-pred-len(m)+1,-pred+1)
    
    fig = plt.figure()
    model_index = sort_list[machine_index][0,2]
    # plot the most similar model
    plt.plot(lib[int(model_index)]['cycle'],lib[int(model_index)]['sequence'],c='r',label='Reference Model')
    # plot the optimal location \tau of testing health index 
    plt.scatter(t,m,label='Health Index')
    # plot another location \tau of testing health index
    #plt.scatter(t-150,m,label='location 2')
    #plt.scatter(t+80,m,label='location 3')

    plt.xlabel('Adjusted Cycles')
    plt.ylabel('Health Index')
    plt.legend(loc='lower left')
    
    
    
    
    
    ### 3rd figure: visualize several life profiles in testing assets versus similar model life profiles
    def arr_rul(rul,t):
        '''
        This function is defined for constructing real life profile of testing asset
        '''
        rul_arr = []
        for i in range(len(t)):
            rul_arr.append(rul+i)
            
        rul_arr.reverse()
        return rul_arr
    
    
    k = 20 # determine how many similar reference models profiles will be visualized for comparison

    for machine_index in np.arange(1,7):
        m1 = test.loc[test['machine'] == machine_index,:]
        t = m1['time']
        rul = label.iloc[machine_index-1,:].values[0]
        
        label_rul = arr_rul(rul,t)
        
        # get the prediction array
        model_pred = sort_list[machine_index-1]
        fig = plt.figure()
        
        # plot several reference models life profiles
        for i in range(k):
            color = (i+1)/10
            pred_arr = arr_rul(model_pred[i,1],t)
            plt.scatter(t,np.array(pred_arr),cmap=color,s=2)
            
        plt.plot(t,label_rul,c='r',linewidth=3,label='ground truth')
        plt.xlabel('time step')
        plt.ylabel('RUL')
        plt.legend()
        fig.savefig(f'figure/top20m{machine_index}modelpred.png')
        
        
        
        ### 4th figure: averaged prediction result versus ground truth
        fig = plt.figure()
        sns.set()
        pred_assemble = np.zeros((len(t),1))
        for i in range(k):
            pred_arr = arr_rul(model_pred[i,1],t)
            pred_arr = np.array(pred_arr)
            pred_assemble = pred_assemble + pred_arr.reshape(-1,1)
        pred_assemble = pred_assemble/k
        plt.scatter(t,pred_assemble,marker='x',s=10,label='prediction')
        plt.plot(t,label_rul,c='r',linewidth=1,label='grond truth')
        plt.legend()     
        fig.savefig(f'figure/top20m{machine_index}aggregatepred.png')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





    