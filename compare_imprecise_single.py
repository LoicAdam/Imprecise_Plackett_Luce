# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:24:45 2021

@author: adamloic
"""

import data.data_ranking as data_ranking
import model.cv_pl as cv
from model.tools_pl import perturbMatrix

import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt


'''Example: compare the contour likelihood algorithm with the abstention algorithm on a dataset.'''
    
def compareImprecisePl(data, test = 0.8, nbR = 5, nbF = 10, p_type = 'miss', p = [0,0.3,0.6], 
                       algo = 'IB', a = [0.95,0.9,0.8,0.7,0.6,0.5],
                       tre = [0.5,0.55,0.6,0.65,0.7], nb = 10000,
                       neighbourhoodType = 'fixed', K = [10,15,20],
                       dirich = [1]):

    completenessImp = np.zeros((len(p), len(a), 3))
    correctnessImp = np.zeros((len(p), len(a), 3))
    completenessAbst = np.zeros((len(p), len(tre), 3))
    correctnessAbst = np.zeros((len(p), len(tre), 3))
    
    for t in range(0, len(p)):
        
        y_pert = perturbMatrix(data[1], p_type, p[t])
        
        imp = cv.modelImprecise(data[0], data[1], y_pert, training_size = test, nbFolds = nbF, 
                                nbRepeats = nbR, algo = algo, neighbourhoodType = neighbourhoodType,
                                K = K[t], alpha = a, nbTraining = nb, dirichCoefs = dirich)
        
        abst = cv.modelAbstention(data[0], data[1], y_pert, training_size = test, nbFolds = nbF, 
                                  nbRepeats = nbR, algo = algo, neighbourhoodType = neighbourhoodType,
                                  K = K[t], threshold = tre)
        
        completenessImp[t,:,0] = imp['Completeness_Lower']
        completenessImp[t,:,1] = imp['Completeness_Avg']
        completenessImp[t,:,2] = imp['Completeness_Upper']
        correctnessImp[t,:,0] = imp['Correctness_Lower']
        correctnessImp[t,:,1] = imp['Correctness_Avg']
        correctnessImp[t,:,2] = imp['Correctness_Upper']
        completenessAbst[t,:,0] = abst['Completeness_Lower']
        completenessAbst[t,:,1] = abst['Completeness_Avg']
        completenessAbst[t,:,2] = abst['Completeness_Upper']
        correctnessAbst[t,:,0] = abst['Correctness_Lower']
        correctnessAbst[t,:,1] = abst['Correctness_Avg']
        correctnessAbst[t,:,2] = abst['Correctness_Upper']
        
    for t in range(0, len(p)):
         
        minCr = np.min(np.hstack((correctnessImp, correctnessAbst)))
        maxCr = np.max(np.hstack((correctnessImp, correctnessAbst)))
        
        colors = [plt.cm.viridis(i) for i in np.linspace(0,0.8,2)]
        fig, ax = plt.subplots(1,1)
    
        ax.plot(completenessImp[t,:,1], correctnessImp[t,:,1], '-o', color = colors[0], label="Likelihood approach")
        ax.plot(completenessImp[t,:,0], correctnessImp[t,:,0], '--', color = colors[0])
        ax.plot(completenessImp[t,:,2], correctnessImp[t,:,2], '--', color = colors[0])
        ax.plot(completenessAbst[t,:,1], correctnessAbst[t,:,1], '-v', color = colors[1], label="Classic abstention")
        ax.plot(completenessAbst[t,:,0], correctnessAbst[t,:,0], '--', color = colors[1])
        ax.plot(completenessAbst[t,:,1], correctnessAbst[t,:,2], '--', color = colors[1])
        ax.set_xlabel('Completeness')
        ax.set_ylabel('Correctness')
        ax.xaxis.set_ticks(np.arange(0, 1.2, 0.2))
        ax.set_ylim(minCr, 1)
        ax.legend(loc="lower left")
        if algo == 'IB':
            title = data[2] + '_' + neighbourhoodType + '_' + p_type + '_' + str(float(p[t]))
        else:
            title = data[2] + '_' + p_type + '_' + str(float(p[t]))
        plt.savefig(title + '.png', dpi = 300)
        tikzplotlib.save(data[2] + '_' + 'correctness.tex', encoding='utf-8')
        
    d = dict()
    d['imp'] = imp
    d['abst'] = abst
    return d

if __name__ == '__main__':
    data = data_ranking.getHousing()
    
    compareImprecisePl(data, test = 0.8, nbR = 1, nbF = 5, p_type = 'swap', p = [0.2], algo = 'IB', 
                       a = np.hstack((0, np.logspace(-15,-2,14), np.linspace(0.1,1,10))), 
                       tre =  np.linspace(0.55,1,10), nb = 100, 
                       neighbourhoodType = 'fixed', K = [20],
                       dirich = np.logspace(0,2,3))