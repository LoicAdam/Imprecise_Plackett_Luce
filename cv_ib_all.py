# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:24:08 2021

@author: adamloic
"""

import data.data_ranking as data_ranking
import model.cv_pl as cv
from model.tools_pl import perturbMatrix

import matplotlib.pyplot as plt
import time
import numpy as np

'''Example: evaluation of the performance of the IB algorithm on all datasets.'''

nbR = 1
nbF = 10
p_type = 'miss'
p = [0,0.3,0.6]
neighbourhoodType = 'distance'
K = [5,10,25,50]
test = 0.8     
      
data_all = data_ranking.getAllIB()
dist_all = np.zeros((len(data_all), len(p), len(K)))
exec_time = np.zeros((len(data_all), len(p), len(K)))
i = 0

for i in range(0, len(data_all)):
    
    data = data_all[i]
    
    for t in range(0,len(p)):    
        
        y_pert = perturbMatrix(data[1], p_type, p[t])
            
        for k in range(0,len(K)):
            
            start = time.perf_counter()
            dist_all[i,t,k] = cv.modelDistance(data[0], data[1], y_pert, 
                                               training_size = test, nbFolds = nbF, nbRepeats = nbR,
                                               algo = 'IB', K = K[k],
                                               neighbourhoodType = neighbourhoodType)['Dist_Avg']
            end = time.perf_counter()
            exec_time[i,t,k] = end - start