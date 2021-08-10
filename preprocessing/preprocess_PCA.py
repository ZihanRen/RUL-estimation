# -*- coding: utf-8 -*-

"""

Created on Thu Feb 27 10:47:36 2020
@author: Zihan Ren - The Pennsylvania State University - Energy and Mineral Engineering Department

Introduction:
    This part of code is mainly used for pre-processing turbofan engine sensors data and labels in both
    training set and testing set. The pre-processing method is PCA transoformation
    
Executive environment and relative APIs:
    
    Python 3.7.6
    Numpy 1.18.1
    pandas 0.25.2
    keras 2.2.4
    scipy 1.3.2
    scikit-learn 0.21.3
    tensorflow 1.14.0

"""

# import packages
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

# main part of code
def main(n_components,phm08_train,phm08_test):
    
    # extract labels from training data
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
    
    # categorize environment paramters into six subgroups using number 0-5 to represent
    def envir_extract(train,test):
        
        '''
        extract environmental data and cluster them into six categories
        '''
        
        envir_data_train = train[['o1','o2','o3']].values
        envir_data_test = test.loc[:,['o1','o2','o3']].values
        
        envir_data = np.vstack((envir_data_train,envir_data_test))
        
        
        kmeans = KMeans(n_clusters = 6, random_state = 0).fit(envir_data)
        label_train = kmeans.predict(envir_data_train)
        label_test = kmeans.predict(envir_data_test)
        
        
        train.insert(3,'envir_label',label_train)
        train = train.drop(['o1','o2','o3'],axis=1 )
        
        test.insert(3,'envir_label',label_test)
        test = test.drop(['o1','o2','o3'],axis=1 )
        
        
        
        return train,test
    
    # one hot encode the environmental parameters
    def envir_todummy(train,test):
        
        '''
        One-Hot-Encoding the categorical environmental data
        '''
        
        dummy_train = pd.get_dummies(train['envir_label'])
        train = train.join(dummy_train)
        train = train.drop(['envir_label'],axis=1)
        
        dummy_test = pd.get_dummies(test['envir_label'])
        test = test.join(dummy_test)
        test = test.drop(['envir_label'],axis=1)
        
        
        return train,test
    
    # scale 
    def scaling(train,test,valid_sensor):
        '''
        scaling all sensor data in each environment individually
        '''
        
        for i in range(6):
        
            scaler = StandardScaler()
            
            data = np.vstack((train.loc[train[i]==1,valid_sensor].values,test.loc[test[i]==1,valid_sensor].values))
            scaler.fit(data)
            
            
            train.loc[train[i]==1,valid_sensor] = scaler.transform(train.loc[train[i]==1,valid_sensor])
            test.loc[test[i]==1,valid_sensor] = scaler.transform(test.loc[test[i]==1,valid_sensor])
    
        
        return train,test
    
    # perform PCA transformation in each individual environment
    def pca_transform(train,test,category,sensor_name):
        
        '''
        Doing PCA transformation on each cluster
        
        One potential pitfall:
            some sensor data is almost the same in one cluster
            If we do the PCA in this domain, this sensor dataset' variance will be disappeared
            
        Original p components = 7
            
        
        '''
        n_com = n_components
        
        
        
        pca = PCA(n_components=n_com)
        pca_data_train = train[ train[category] == 1 ]
        pca_data_test = test[ test[category] == 1 ]
        pca_data = np.vstack((pca_data_train[sensor_name].values,pca_data_test[sensor_name].values))
        
        pca.fit(pca_data)
        
        
        pca_sensor_train = pca.transform(pca_data_train[sensor_name].values)
        pca_sensor_test = pca.transform(pca_data_test[sensor_name].values)
        
        num_components = n_com
        
        component_name = []
        
        for i in range(num_components):
            component_name.append('pc'+str(i+1))
        
        
        pca_sensor_train = pd.DataFrame(pca_sensor_train,columns=component_name)
        pca_sensor_test = pd.DataFrame(pca_sensor_test,columns=component_name)
        
        origin_train = pca_data_train[['machine','time','rul',0,1,2,3,4,5]]
        origin_test = pca_data_test[['machine','time',0,1,2,3,4,5]]
        
        concat_train = np.hstack((origin_train.values,pca_sensor_train.values))
        concat_test = np.hstack((origin_test.values,pca_sensor_test.values))
        
        train_name = ['machine','time','rul',0,1,2,3,4,5] + component_name
        test_name = ['machine','time',0,1,2,3,4,5] + component_name
        
        output_train = pd.DataFrame(concat_train,columns=train_name)
        output_test = pd.DataFrame(concat_test,columns=test_name)
        
                
        return output_train,output_test
    
    # rearrange the PCA pre-processed data set
    def rearrange_pca(data_pca,pca_data):
        '''
        Input:
            numpy array of all vstack pca data
            one pca_data dataframe to extract column names
        Output:
            rearranged pca data frame for all. Sorted by machine and time step
        
        '''
        
        pca_name = list(pca_data.columns.values)
        data_pca = pd.DataFrame(data_pca,columns=pca_name)
        
        
        
        machine_data1 = data_pca[data_pca['machine']==1]
        machine_data1 = machine_data1.sort_values('time')
        
        
        machine_data2 = data_pca[data_pca['machine']==2]
        machine_data2 = machine_data2.sort_values('time')
        
        machine_datas = [machine_data1,machine_data2]
        
        rearrange_data = pd.concat(machine_datas,axis=0,ignore_index=True)
        
        
        for i in range(3,len(np.unique(data_pca.loc[:,'machine'].values))+1):
            
            machine_data = data_pca[data_pca['machine']==i]
            machine_data = machine_data.sort_values('time')
            machine_datas = [rearrange_data,machine_data]
            rearrange_data = pd.concat(machine_datas,axis=0,ignore_index=True)
            
            
        return rearrange_data
    
    
    
    ### perform pre-process using above functions
    cat_sensor = ['s1','s5','s6','s10','s16','s17','s18','s19']
    # selected valid sensors for further processing
    valid_sensor = [ 's2', 's3', 's4', 's7', 's8', 's9',
           's11', 's12', 's13', 's14', 's15', 's20', 's21']
    
    phm08_train = phm08_train.drop(cat_sensor,axis=1)
    phm08_test = phm08_test.drop(cat_sensor,axis=1)
    
    
    phm08_train = rulextract(phm08_train)
    phm08_train,phm08_test = envir_extract(phm08_train,phm08_test)
    phm08_train,phm08_test = envir_todummy(phm08_train,phm08_test)
    
    phm08_train,phm08_test = scaling(phm08_train,phm08_test,valid_sensor)
    
    # conducting PCA transformation on each category dataset
    train_pca_data,test_pca_data = pca_transform(phm08_train,phm08_test,0,valid_sensor)
    train_pca_data1,test_pca_data1 = pca_transform(phm08_train,phm08_test,1,valid_sensor)
    train_pca_data2,test_pca_data2 = pca_transform(phm08_train,phm08_test,2,valid_sensor)
    train_pca_data3,test_pca_data3 = pca_transform(phm08_train,phm08_test,3,valid_sensor)
    train_pca_data4,test_pca_data4 = pca_transform(phm08_train,phm08_test,4,valid_sensor)
    train_pca_data5,test_pca_data5 = pca_transform(phm08_train,phm08_test,5,valid_sensor)
    
    
    
    
    
    
    data_pca_train = np.vstack((train_pca_data.values,train_pca_data1.values))
    data_pca_train = np.vstack((data_pca_train,train_pca_data2.values))
    data_pca_train = np.vstack((data_pca_train,train_pca_data3.values))
    data_pca_train = np.vstack((data_pca_train,train_pca_data4.values))
    data_pca_train = np.vstack((data_pca_train,train_pca_data5.values))
    
    
    data_pca_test = np.vstack((test_pca_data.values,test_pca_data1.values))
    data_pca_test = np.vstack((data_pca_test,test_pca_data2.values))
    data_pca_test = np.vstack((data_pca_test,test_pca_data3.values))
    data_pca_test = np.vstack((data_pca_test,test_pca_data4.values))
    data_pca_test = np.vstack((data_pca_test,test_pca_data5.values))
    
    
    
    
    # rearrange the transformed values
    rearrange_data_train = rearrange_pca(data_pca_train, train_pca_data)
    rearrange_data_test = rearrange_pca(data_pca_test,test_pca_data)

    return rearrange_data_train,rearrange_data_test


if __name__ == '__main__':
    
    column_name = ['machine','time','o1','o2','o3']
    sensor_name = []
    for i in range(21):
        column_name.append('s'+str(i+1))
        sensor_name.append('s'+str(i+1))
    del i
    
    phm08_train = pd.read_csv(r'kaggle\CMaps\train_FD002.txt',
                         delim_whitespace=True,header=None,names=column_name)
    phm08_test = pd.read_csv(r'kaggle\CMaps\train_FD002.txt',
                         delim_whitespace=True,header=None,names=column_name)
    train,test = main(7,phm08_train,phm08_test)
    
    

    


    








