from model.IB_pl import IBPlLearner
from model.GLM_pl import GLMPlLearner

import numpy as np

def createMissingLabels(y, p = 0.05):
    """
    For each label of a ranking, a biased coin (with a probability of p) is flipped
    to know if a label is deleted or not. p * 100% of labels are deleted in
    average.
    """
    
    #Set some labels to zero.
    biasedCoin = np.random.random(size=y.shape)
    mask = np.where(biasedCoin >= p, 1, 0)
    newRanking = np.multiply(y, mask)
    
    return newRanking

def swappingLabels(y, p = 0.05):
    """
    Swap randomly some labels inside rankings in a ranking matrix. Only neighbour
    pairs of label can be swapped. p*100 is the number of neighbour pairs that 
    are swapped in average.
    """
    
    #Obtain a matrix swappingInd to tell for each line which neighbour labels shall be
    #swapped. For example, one in the first columns means label 0 and 1 will be swapped.
    swapping = np.random.rand(y.shape[0],y.shape[1]-1)
    swappingInd = np.where(swapping <= p, 1, 0)
    
    #For loops are not fast, but the swapping should not be done often (once per
    #dataset).
    for i in range(0, y.shape[0]):
        
        #Check if there are swaps to do:
        if np.count_nonzero(swappingInd[i]) == 0:
            continue
        
        #Idea: the swaps are permuted randomly so it's not always the first labels
        #which are swapped.
        swaps = np.where(swappingInd[i] > 0)[0]
        permuted_swaps = np.random.permutation(swaps)
        
        for j in permuted_swaps:
            
            tmp = y[i,j]
            y[i,j] = y[i,j+1]
            y[i,j+1] = tmp
        
    return y

def perturbMatrix(y, p_type = 'miss', p = 0.05):
    """
    Perturb a matrix by swapping some labels or add missing labels.
    """
    
    if p_type == 'swap':
        y = swappingLabels(y, p)
        
    elif p_type == 'miss':
        y = createMissingLabels(y, p)
        
    else:
        raise ValueError('Unknown method to perturb matrix.')
        
    return y

def algoChoice(algo, nbTypesLabels = 10, neighbourhoodType = 'fixed', K=15, 
               opti='MM'):
    """
    Prepare the objets corresponding to specific algorithms (instanced-based, GLM...)
    used to learn rankings.
    """
    
    if algo == 'IB':
        pl_learner = IBPlLearner(neighbourhoodType, K, opti, nbTypesLabels)
    elif algo == 'GLM':
        pl_learner = GLMPlLearner(nbTypesLabels)
    else:
        raise ValueError('Unknown learning algorithm was given.')
        
    return pl_learner