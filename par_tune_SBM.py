# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:50:45 2020
For Parameter Tuning 
@author: zur74
"""

import bayes_cluster as b_clus
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import random





def main(machine_index,top_c):
    
    

    def range_tau(model_length,m1):
    
        # define the parameter of model time attribute
        r = len(m1) # number of cycles in test sequence
        tau_min = 0 # minimum of remaining cycles 
        tau_max = model_length - r # maximum remaining cycles
        tau_range = np.arange(tau_min,tau_max+1)
    
        return tau_range
    
    def sequence_output(s1,location,r):
    
        '''
        This function is mainly used for outputing the model sequence for comparing with test sequence
        s1: selected model sequence
        location: remaining cycles indicating the location of model sequence
        m1: the output sequence length
        '''
        # get the beginning and end time in cycle
        cycle_begin = -location - r+1
        cycle_end = -location -r + r
        
        # construct the comparing sequence in model
        s_temp = s1[(s1['cycle'] >= cycle_begin) & (s1['cycle'] <= cycle_end)]
        
        return s_temp.loc[:,'sequence'].values,s_temp.loc[:,'top'].values,s_temp.loc[:,'bottom'].values
    
    def sequence_output_cluster(s1,location,r):
    
        '''
        This function is mainly used for outputing the model sequence for comparing with test sequence
        s1: selected model sequence
        location: remaining cycles indicating the location of model sequence
        m1: the output sequence length
        '''
        # get the beginning and end time in cycle
        cycle_begin = -location - r +1
        cycle_end = -location -r + r
        
        # construct the comparing sequence in model
        s_temp = s1[(s1['cycle'] >= cycle_begin) & (s1['cycle'] <= cycle_end)]
        
        return s_temp.loc[:,'sequence'].values,s_temp.loc[:,'top'].values,s_temp.loc[:,'bottom'].values
    
    
    def sim_score_area(vec1,vec2,vec1_top,vec1_bottom):
        
        '''
        Distance between health index and nearest boundary
        '''
        
        if len(vec2[(vec2>=vec1_bottom) & (vec2<=vec1_top)]) == len(vec2):
            score = 0
        else:
            if len(vec2[vec2>vec1_top]) > 0:
                diff1 = (vec2[vec2>vec1_top].reshape(-1,1) - vec1_top[vec1_top<vec2].reshape(-1,1)) **2
            else:
                diff1 = 0
            if len(vec2[vec2<vec1_bottom]) > 0:
                diff2 = (vec2[vec2<vec1_bottom].reshape(-1,1) - vec1_bottom[vec1_bottom>vec2].reshape(-1,1))**2
            else:
                diff2 = 0
        
            score = np.sum(diff1) + np.sum(diff2)
        
        return score
    
    def sim_score(vec1,vec2):
        diff = (vec1.reshape(-1,1) - vec2.reshape(-1,1))**2 
        return np.sum(diff)
        
    def score_sort(model_result,lib):
        
        # get the most similar models and rank them
        model_stack = np.hstack(( model_result,np.arange(0,len(lib)).reshape(-1,1) ))
        model_sort = model_stack[model_stack[:,0].argsort()]
        
        return model_sort
    
    def weight_predict(model_sort,k,sequence_len):
        
        '''
        get the k nearest prediction
        k should be even number
        w = (0.5 + 0.5^2 + 0.5^3 + ... + 0.5^k-1) + 0.5^k-1
        '''
        
        # filter the noise
        
        select_matrix = sequence_len + model_sort[0:k,1]
        select_matrix = select_matrix[(select_matrix<310)&(select_matrix>120) ]
        
        if len(select_matrix) < 5:
            prediction_matrix = model_sort[0:k,1]
            
        else:
        
            prediction_matrix = select_matrix - sequence_len
        
        return np.mean(prediction_matrix)
    
    def find_cluster_test(seq_rep,m1,lib,top_output):
        tau_array = range_tau(len(lib[0]),m1)
        
        r = len(m1)
        cluster_score = []
        for cluster in range(len(seq_rep)):
            seq_cluster = seq_rep[cluster,:].reshape(-1,1)
            # construct the data frame
            seq_cluster = np.hstack((seq_cluster,lib[0].loc[:,'cycle'].values.reshape(-1,1)))
            seq_cluster = pd.DataFrame(seq_cluster,columns = ['sequence','cycle'])
            score = []
            for location in tau_array:
                # get the beginning and end time in cycle
                cycle_begin = -location - r +1
                cycle_end = -location -r + r
                
                # construct the comparing sequence in model
                s_temp = seq_cluster[(seq_cluster['cycle'] >= cycle_begin) & (seq_cluster['cycle'] <= cycle_end)]
                distance_s = sim_score(s_temp.loc[:,'sequence'].values.reshape(-1,1),
                                       m1.loc[:,'smooth'].values.reshape(-1,1))
                score.append(distance_s)
            cluster_score.append(min(score))
            cluster_sort_index = np.argsort(np.array(cluster_score))
        
        return cluster_sort_index[0:top_output]
    
    def output_cluster_lib(cluster_label,cluster_attribute,lib):
    
        # determine which cluster that the test sequence belong to
        cluster_label = list(cluster_label)
        index_list = []
        
        for j in cluster_attribute:
            for index,value in enumerate(cluster_label):
                if value == j:
                    index_list.append(index)
            
        lib_clus = []
        for i in index_list:
            lib_clus.append(lib[i])
            
        return lib_clus
    
    
    
    # test,lib,seq_rep,cluster_label are all global variables since this is the final script
    
    # make a prediction on one unit
    m1 = test[test['machine'] == machine_index]
    tau_array = range_tau(len(lib[0]),m1)
    
    cluster_attribute = find_cluster_test(seq_rep, m1, lib,top_c)
    lib_clus = output_cluster_lib(cluster_label, cluster_attribute, lib)
    
    
    
    # construct the matrix of output 
    # the first column has attribute of score and second column has attribute of tau - remaining useful life
    model_result = np.zeros((len(lib_clus),2))
    
    for model in range(len(lib_clus)):
        # looping models in model library
        s1 = lib_clus[model]
        # get the collection of score given different tau
        score = []
        for location in tau_array:
            sequence_model,top,bottom = sequence_output(s1,location,len(m1))
            distance = sim_score_area(sequence_model,m1.loc[:,'smooth'].values,top,bottom)
            score.append(distance)    
        
        
        # find the best score with lowest value and corresponding tau
        best_score = min(score)
        best_tau = tau_array[score.index(min(score))]
        
        # get the result of current machine similarity condition - tau and corresponding score
        model_result[model,0] = best_score
        model_result[model,1] = best_tau
    
    # normalize the model result
    model_result[:,0] = model_result[:,0]/np.sum(model_result[:,0])
    
    # sort the model and output sorted result and sorted score
    model_sort = score_sort(model_result,lib_clus)
    rul = weight_predict(model_sort,20,len(m1))
    
    return rul,model_sort,cluster_attribute,lib_clus



def rulextract(data):
    
    '''
    return the array of remaining useful life
    '''
    
    label_sum = np.zeros((1,1))
    
    for i in np.unique(data['machine'].values):
        values = data[data['machine'] == i]['time']
        last_time = np.max(values)
        label = last_time - values
        label_sum = np.vstack((label_sum,label.values.reshape(-1,1)))
    label_sum = np.delete(label_sum,0,axis = 0)
    
    data.insert(2,'rul',label_sum)
    
    return data






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
    
    
    
    mae_arr = []
    mse_arr = []
    par_arr = []
    time_arr = []
    machine_num = np.unique(phm08_test['machine']).tolist()
    sample_num = 10
    machine_num = machine_num[0:sample_num]
    for num_c in np.arange(3,14):
        for top_c in np.arange(2,4):
            
            cluster_label,seq_rep,lib,train,test = b_clus.main(num_c,phm08_train,phm08_test)
            
            

        
            # begin prediction
            start = time.perf_counter()
            rul_arr = []

            print('prediction begins ...')
            for machine_index in machine_num:
                print(f'   Machine {machine_index}:')
                print('...prediction begins...')
                rul,model_sort,rep,lib_clus = main(machine_index,top_c)
                rul_arr.append(rul)
                print('...prediction complete...\n')
            
            finish = time.perf_counter()
            
            par = (num_c,top_c)
            par_arr.append(par)
            
            
            mae = np.mean(abs(label.iloc[0:sample_num].values.reshape(-1,1) - 
                                  np.array(rul_arr).reshape(-1,1)))
            mae_arr.append(mae)
            mse = np.mean((label.iloc[0:sample_num].values.reshape(-1,1) - 
                                  np.array(rul_arr).reshape(-1,1))**2)
            mse_arr.append(mse)
            # seconds
            compute_time = round(finish-start,2)
            time_arr.append(compute_time)
            
    
    
    
    

    
    
    
    

    
    
    

    
    
    





