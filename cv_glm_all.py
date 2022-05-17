# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:24:44 2021

@author: adamloic
"""

import data.data_ranking as data_ranking
import model.cv_pl as cv
from model.tools_pl import perturbMatrix

import numpy as np
import matplotlib.pyplot as plt
import time

'''Example: evaluation of the performance of the GLM algorithm on all datasets.'''
    
test = 0.8
nbR = 1
nbF = 10
p_type = 'miss'
p = [0,0.3,0.6]
    
data_all = data_ranking.getAllGLM()
del(data_all[2])
del(data_all[1])
dist_all = np.zeros((len(data_all),len(p),3))
exec_time = np.zeros((len(data_all), len(p),3))
i = 0

for i in range(0, len(data_all)):
    
    data = data_all[i]
    
    for t in range(0,len(p)):    
        
        y_pert = perturbMatrix(data[1], p_type, p[t])
        
        start = time.perf_counter()
        res = cv.modelDistance(data[0], data[1], y_pert, training_size = test, 
                               nbFolds = nbF, nbRepeats = nbR, algo = 'GLM')
        end = time.perf_counter()
        exec_time[i,t] = end - start

        dist_all[i,t,0] = res['Dist_Lower']
        dist_all[i,t,1] = res['Dist_Avg']
        dist_all[i,t,2] = res['Dist_Upper']
        
    plt.clf()
    plt.plot(p, dist_all[i,:,0], '--r')
    plt.plot(p, dist_all[i,:,1], '-r')
    plt.plot(p, dist_all[i,:,2], '--r')
    plt.xlabel('Missing label percentage')
    plt.ylabel('Distance')
    plt.title(data[2])