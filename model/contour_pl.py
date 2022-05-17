from model.abstention_pl import preferenceMatrix, rankingAbstention
from model.IB_pl import IBPlLearner
from model.GLM_pl import GLMPlLearner

import numpy as np

def overAlpha(l, v, alpha):
    """
    Select points with a likelihood ratio over alpha.
    """

    l_sorted = np.argwhere(l >= alpha)
    l_selected = l[l >= alpha]
    
    v_selected = v[l_sorted]
    v_selected = np.squeeze(v_selected)
    return v_selected, l_selected

def generateNormL(v_opt, neighbours, nbTypesLabels,
                  dirichCoefs = [1,10], nbTraining = 10000, tol = 10e-15,
                  learner = None, algo = 'IB', X = []):
    """
    Generation coeffecients of a PL related to a dataset and then get their
    loglikelihood to then have their likelihood ratio."""
    
    #If no learner: we suppose we want to learn from the whole dataset.
    if learner == None:
        learner = IBPlLearner('fixed', neighbours.shape[0], 'MM', nbTypesLabels)
        learner.fit([], neighbours)
        algo = 'IB'
      
    v = [v_opt]                   
    #Zeros are replaced by a number close to zero to be able to make
    #predictions.
    v_generate = np.where(v_opt < tol, tol, v_opt)
    v_generate = v_generate/np.sum(v_generate)
    
    #Generating random v.
    for coef in dirichCoefs:
        v = np.vstack((v, np.random.dirichlet(alpha = coef * v_generate, size = nbTraining)))
    
    v = np.where(v < tol, tol, v)
    v = v/np.sum(v, axis=1)[:,None]
    
    l = np.zeros(nbTraining)
    
    '''params = markovParameters(neighbours, nbTypesLabels)        
    W = params['W']
    C = params['C']
    #Remove lines and columns with zeros.
    W = W.T[~np.all(W == 0, axis=0)].T
    #W = W[~np.all(W == 0, axis=1)]
    C = C.T[~np.all(C == 0, axis=0)].T
    #C = C[~np.all(C == 0, axis=1)]'''

    #Calculate the likelihood for each v
    if algo == 'IB':
        l = learner.loglikelihood(v, neighbours)
    else:
        #Transform into neighbours
        y = np.argsort(v, axis = 1).T[::-1].T + 1
        fullX = np.repeat(X[np.newaxis, :], y.shape[0], axis=0)
        l = learner.loglikelihood(fullX, y)
      
    l_opt = np.max(l)  
    l_norm = np.exp(l - l_opt)
    
    return l_norm, v

def impreciseOrder(X_train, y_train, X_test, nbTypesLabels, 
                   alpha = [0.8,0.9], nbTraining = 10000, algo = 'IB', 
                   neighbourhoodType = 'fixed', K=10, opti='MM',
                   dirichCoefs = [1]):
    """
    Find the MLE of the plackett-luce model and then generate points to 
    have the full loglikelihood. From that, some points with the highest
    likelihood are selected to do some imprecise loglikelihood estimation.
    Get the order matrices in the case of imprecise learning on a dataset.
    """
    
    v_opt = 0
    learner = None
    
    if algo == 'IB':
        learner = IBPlLearner(neighbourhoodType, K, opti, nbTypesLabels)
    elif algo == 'GLM':
        learner = GLMPlLearner(nbTypesLabels)
    else:
        raise ValueError('Unknown learning algorithm was given.')

    prefMat = np.zeros((X_test.shape[0], len(alpha), nbTypesLabels, nbTypesLabels))
    nbSelectedPoints = np.zeros((len(alpha), X_test.shape[0]))
    
    tol = 10e-15
    threshold = 0.5
    
    learner.fit(X_train, y_train)
    v_opt = learner.predict(X_test)[0]
        
    for i in range(0, X_test.shape[0]):
        
        if algo == 'IB':
            element = X_test[i,:]
            neighbour_ranking = learner.findClosestNeighbours(element)
        elif algo == 'GLM':
            neighbour_ranking = y_train
        else:
            raise ValueError('Unknown learning algorithm was given.')
        
        l_norm, v = generateNormL(v_opt[i,:], neighbour_ranking, nbTypesLabels,
                                       dirichCoefs, nbTraining, tol, learner,
                                       algo, X_test[i,:])
        
        #Add a dimension to the array if only one dimension, to avoid errors.
        if v.ndim == 1:
            v = v[None,:]
        
        k = 0
        for a in alpha:
            
            v_select, l_selected = overAlpha(l_norm, v, a)
            
            #Alpha = 1 and multiple selected: technically impossible.
            #Only due to computational errors. We take only one.
            if a == 1 and v_select.ndim > 1:
                v_select = v_select[0,:]
                l_selected = l_selected[0]
            
            #Add a dimension to the array if only one dimension, to avoid errors.
            if v_select.ndim == 1:
                v_select = v_select[None,:]
            
            mat = np.zeros((v_select.shape[0], nbTypesLabels, nbTypesLabels))
            absMat = np.zeros((v_select.shape[0], nbTypesLabels, nbTypesLabels))
    
            mat = preferenceMatrix(v_select, nbTypesLabels)
            absMat = rankingAbstention(mat, threshold)
            
            prefMat[i,k,:,:] = np.sum(absMat, axis = 0)/v_select.shape[0]
            nbSelectedPoints[k,i] = v_select.shape[0]
            
            k = k+1
                    
    return prefMat, nbSelectedPoints