# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:24:44 2021

@author: adamloic
"""

import data.data_ranking as data_ranking
import model.cv_pl as cv
from model.tools_pl import perturbMatrix

import matplotlib.pyplot as plt
import numpy as np

'''Example: evaluation of the performance on a dataset with missing labels.'''

test = 0.8 
nbR = 5
nbF = 10
algo = 'IB'
miss = [0,0.3,0.6]
neighbourhoodType = 'fixed'
K = [10,15,20]
data = data_ranking.getIris()
        
dist = np.zeros((len(miss),3))

if len(K) != len(miss):
    K = np.repeat(K[0], len(miss))

for t in range(0, len(miss)):
    
    y_pert = perturbMatrix(data[1], 'miss', miss[t])
        
    res = cv.modelDistance(data[0], data[1], y_pert,
                           training_size = test, nbRepeats = nbR, nbFolds = nbF, 
                           algo = algo, neighbourhoodType = neighbourhoodType, K = K[t])
    
    dist[t,0] = res['Dist_Lower']
    dist[t,1] = res['Dist_Avg']
    dist[t,2] = res['Dist_Upper']
    
plt.clf()
plt.plot(miss, dist[:,0], '--r')
plt.plot(miss, dist[:,1], '-r')
plt.plot(miss, dist[:,2], '--r')
plt.xlabel('Missing label percentage')
plt.ylabel('Distance')