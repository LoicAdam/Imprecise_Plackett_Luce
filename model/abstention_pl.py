import numpy as np

def preferenceMatrix(strength, nbTypesLabels):
    """
    Preference matrix: to know if i is preferred to y. > 0.5 means yes, < no.
    Goes from 0 to 1.
    Accept an array of matrices.
    """

    if strength.ndim == 1:
        strength = strength[None,:]
    nbRep = strength.shape[0]
    
    #Column i : strength i
    matPrefered = np.repeat(strength[:, np.newaxis, :], nbTypesLabels, axis = 1)
    #Line i : strength i.
    matPrefer = np.transpose(matPrefered, ((0,2,1)))
    
    #Get the preferences with Bradley Terry model.
    #If division by zero : return zero.
    matDem = np.add(matPrefer, matPrefered)
    prefMat = np.divide(matPrefer, matDem, out = np.zeros_like(matPrefer),
                        where = matDem != 0)
    
    #An element is not preferred to itself.
    mask = np.eye(prefMat.shape[1], dtype=bool)
    multiMask = np.repeat(mask[np.newaxis,: ,: ], nbRep, axis = 0)
    prefMat[multiMask] = 0
            
    return prefMat

def breakTies(prefMat):
    """
    It is possible in some cases i > j with a probability of 0.5 and
    j > i with a probability of 0.5.
    When a threshold of 0.5 is applied, this is problem. These ties
    have to be broken.
    """
    
    if prefMat.ndim == 2:
        prefMat = prefMat[None,:,:]
        
    for i in range(0, prefMat.shape[0]):
        
        ties = np.where(prefMat[i,:,:] == 0.5)
                  
        if ties[0].size == 0:
            continue
        
        rows, cols = ties
        for t in range(0, rows.size):
            
            row = rows[t]
            col = cols[t]
            
            #If the tie is already broken: pass.
            if prefMat[i,row,col] == 0 or prefMat[i,col,row] == 0:
                continue
            
            #Break randomly the tie.
            else:
                tieBreak = np.random.randn(1)
                if tieBreak < 0.5:
                    prefMat[i,col,row] = 0
                else:
                    prefMat[i,row,col] = 0
        
    return prefMat

def rankingAbstention(prefMat, threshold):
    """
    Thresholding approach: when < threshold, the preferrence is set to 0.
    Otherwise 1.
    """

    #Check the value of the threshold is correct [0.5, 1]
    if threshold > 1 or threshold < 0.5:
        raise ValueError('Wrong value for the threshold.')
        
    '''if threshold == 0.5:
        prefMat = self.breakTies(prefMat)'''
    
    #1 if the value is over the threshold. 0 otherwise.
    abstMat = np.where(prefMat >= threshold, 1, 0)
    return abstMat