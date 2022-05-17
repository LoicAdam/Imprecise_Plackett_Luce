import itertools
import numpy as np

def rankingToOrderMatrix(rankings, nbTypesLabels):
    """
    Transform a ranking into an order matrix.
    """
    
    nbRankings = rankings.shape[0]
    nbRanked = rankings.shape[1]
    matrices = -np.ones((nbRankings, nbTypesLabels, nbTypesLabels))
    
    for r in range(0,nbRankings):
        for i in range(0, nbRanked):
            
            ele = rankings[r,i]-1
            
            matrices[r,ele,:] = np.where(matrices[r,ele,:] == -1, 1, 0)
            matrices[r,:,ele] = np.where(matrices[r,:,ele] == -1, 0, 1)
            
        np.fill_diagonal(matrices[r,:,:], 0)
        
    return matrices

def kendalltau(r1, r2):
    """
    Calculate the kendall tau distance of two rankings. Ties are not considered.
    """
        
    row = r1.shape[0]
    distances = np.zeros(row)

    for i in range(row):
        
        r1_row = np.argsort(r1[i,:])+1
        r2_row = np.argsort(r2[i,:])+1
        M = np.count_nonzero(r1_row)
        pairs = itertools.combinations(range(0, M), 2)
        C = 0
        D = 0
    
        for x, y in pairs:
            r1_pair = r1_row[x] - r1_row[y]
            r2_pair = r2_row[x] - r2_row[y]
    
            if (r1_pair * r2_pair > 0):
                C += 1
            elif (r1_pair * r2_pair < 0):
                D += 1
        
        distances[i] = np.divide(C-D, ((M*(M-1))/2))
        
    return distances

def correctness(o1,o2):
    """
    Calculate the correctness of an order matrix. Ties are not considered.
    """
        
    nbOrders = o1.shape[0]
    correctness = np.zeros(nbOrders)

    for i in range(0,nbOrders):
        
        M = o1.shape[1]
        pairs = itertools.combinations(range(0, M), 2)
        C = 0
        D = 0
    
        for x, y in pairs:
    
            if ((o1[i,x,y] == 1 and o2[i,x,y] == 1) or (o1[i,y,x] == 1 and o2[i,y,x] == 1)):
                C += 1
            elif ((o1[i,x,y] == 1 and o2[i,y,x] == 1) or (o1[i,y,x] == 1 and o2[i,x,y] == 1)):
                D += 1
        
        if (C+D) == 0:
            correctness[i] = 1
        else:
            correctness[i] = np.divide(C-D, C+D)
        
    return correctness

def completeness(o1,o2):
    """
    Calculate the completeness of an order matrix. Ties are not considered.
    """
        
    nbOrders = o1.shape[0]
    completeness = np.zeros(nbOrders)
    for i in range(0, nbOrders):
        
        M = o1.shape[1]
        pairs = itertools.combinations(range(0, M), 2)
        C = 0
        D = 0
        
        for x, y in pairs:
    
            if ((o1[i,x,y] == 1 and o2[i,x,y] == 1) or (o1[i,y,x] == 1 and o2[i,y,x] == 1)):
                C += 1
            elif ((o1[i,x,y] == 1 and o2[i,y,x] == 1) or (o1[i,y,x] == 1 and o2[i,x,y] == 1)):
                D += 1
                
        completeness[i] = np.divide(C+D, ((M*(M-1))/2))
        
    return completeness
