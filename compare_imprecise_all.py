# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:24:45 2021

@author: adamloic
"""

import data.data_ranking as data_ranking
from compare_imprecise_single import compareImprecisePl

import numpy as np

'''Example: compare the contour likelihood algorithm with the abstention algorithm on multiple datasets.'''
    
nbR = 5
nbF = 10
p_type = 'swap'
algo = 'IB'
a = np.hstack((0, np.logspace(-15,-2,14), np.linspace(0.1,1,10)))
nb = 200 
tre =  np.linspace(0.5,1,11)
neighbourhoodType = 'distance'
diri = np.logspace(0,10,11)
minimumK = False
test = 0.8

if p_type == 'miss':
    p = [0,0.3,0.6]
else:
    p = [0,0.2,0.4,0.6]

if algo == 'IB':
    
    data_all = data_ranking.getAllIB()
    K_fixedMiss = [[5,20,20],[15,15,20],[5,5,20],[5,10,15],[5,10,20],[5,10,20],
                   [15,15,20],[5,5,10],[10,10,20],[10,15,20]]
    K_fixedSwap = [[10,15,20,20],[15,15,20,20],[5,5,20,20],[5,10,15,20],
                   [5,15,20,20],[5,10,20,20],[20,20,20,20],[5,5,5,5],
                   [10,10,15,15],[10,10,15,15]]
    
    K_varMiss = [[15,15,25],[10,10,10],[10,25,25],[5,10,10],[15,15,25],
                 [5,5,15],[5,5,10],[15,15,25],[25,25,25],[15,15,15]]
    K_varSwap = [[10,25,25,25],[15,15,25,25],[5,5,20,20],[5,10,10,15],
                 [5,25,25,25],[5,15,25,25],[10,10,25,25],[10,15,15,15],
                 [25,25,25,25],[10,10,15,15]]
    
    res_fixed = []
    res_var = []
    
    for i in range(0, len(data_all)):
        
        data = data_all[i]
        if p_type == 'miss':
            if minimumK == True:
                Kf = np.repeat(5,3)
                Kv = np.repeat(5,3)
            else:
                Kf = K_fixedMiss[i]
                Kv = K_varMiss[i]
                
        else:
            if minimumK == True:
                Kf = np.repeat(5,4)
                Kv = np.repeat(5,4)
            else:
                Kf = K_fixedSwap[i]
                Kv = K_varSwap[i]
        
        if neighbourhoodType == 'fixed':
            res_fixed.append(compareImprecisePl(data, test, nbR, nbF , p_type, p, algo, 
                                                a, tre, nb, neighbourhoodType = 'fixed',
                                                K = Kf, dirich = diri))
        if neighbourhoodType == 'distance':
            res_var.append(compareImprecisePl(data, test, nbR, nbF, p_type, p, algo, 
                                                a, tre, nb, neighbourhoodType = 'distance',
                                                K = Kv, dirich = diri))
       
else:
    
    data_all = data_ranking.getAllGLM()
    res = []
    
    for i in range(0, len(data_all)):
        data = data_all[i]
        res.append(compareImprecisePl(data, nbR, nbF, p_type, p, algo, 
                                      a, tre, nb))
        
        