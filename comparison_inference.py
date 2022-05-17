from model.opti_pl import coefs, markovParameters
from model.contour_pl import overAlpha
from model.abstention_pl import rankingAbstention, preferenceMatrix
from model.IB_pl import IBPlLearner
import data.data_ranking as data_ranking

import numpy as np
import choix

def likelihood(neighbours, nbTypesLabels, v):
    
    ib_pl = IBPlLearner('fixed', 5, 'MM', nbTypesLabels)
    l = ib_pl.loglikelihood(v, neighbours)
    return l
    
def contour(neighbours, nbTypesLabels, nbTraining, dirichCoefs, nbAlpha):
    
    #Optimal coefficients.
    v_opt = coefs(neighbours, nbTypesLabels, 'MM')['strength']

    alpha = np.linspace(1,0,nbAlpha)

    #Generate a number of strenghs according to a dirichlet distribution.
    v = v_opt
    for coef in dirichCoefs:
        v = np.vstack((v, np.random.dirichlet(alpha = coef * v_opt, size = nbTraining)))
        
    #Get their loglikelihood
    l = likelihood(neighbours, nbTypesLabels, v)
    
    #Get the contour likelihood
    l_contour = np.exp(l - l[0])
    
    #Get the different orders according to beta cuts
    voteMat = np.zeros((nbAlpha, nbTypesLabels, nbTypesLabels))
    orderMat = np.zeros((nbAlpha, nbTypesLabels, nbTypesLabels))
    k = 0
    for a in alpha:
        
        v_select, l_selected = overAlpha(l_contour, v, a)
        
        #Add a dimension to the array if only one dimension, to avoid errors.
        if v_select.ndim == 1:
            v_select = v_select[None,:]
        
        mat = np.zeros((v_select.shape[0], nbTypesLabels, nbTypesLabels))
        absMat = np.zeros((v_select.shape[0], nbTypesLabels, nbTypesLabels))
    
        mat = preferenceMatrix(v_select, nbTypesLabels)
        absMat = rankingAbstention(mat, 0.5)
        
        voteMat[k,:,:] = np.sum(absMat, axis = 0)/v_select.shape[0]
        orderMat[k,:,:] = np.where(voteMat[k,:,:] == 1, 1, 0)
        
        k = k+1
        
    return v, l_contour, voteMat, orderMat
    
#Synthetic data set.
nbTypesLabels = 5
pl_parameters = np.random.rand(nbTypesLabels)
pl_parameters = np.divide(pl_parameters, np.sum(pl_parameters))
nb_rankings = 1000
size_rankings = 5
neighbours = np.asarray(choix.generate_rankings(pl_parameters, nb_rankings, size_rankings)).astype(int)+1

nbTraining = 1000
dirichCoefs = [100,1000,10000]
nbAlpha = 11
alpha = np.linspace(1,0,nbAlpha)
v, l_contour, voteMat, orderMat = contour(neighbours, nbTypesLabels, nbTraining, dirichCoefs, nbAlpha)
