# RUL-estimation
RUL estimation of turbofans using C-MAPSS data set.  
Data source: NASA PCoE repository  
url: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan.  
Reference paper: "Application of Data Driven Machine Learning Methodologies for Predicting Reamining Useful Life of Equipment"  
Reference paper url: https://etda.libraries.psu.edu/catalog/18344zur74. 


Scripts intro:  
 rnn.py: implement deep recurrent neural network on data set.  
 ann.py: implement deep feedforward neural network on data set.  
 SBM.py: implement classic similarity based method on data set.  
 extendSBM.py: develop an extended SBM method on data set.  
  
 par_tune_SBM.py: finding optimal cluster groups on data set when using extended SBM.  
