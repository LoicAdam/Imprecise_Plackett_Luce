import numpy as np
import time
 
def coefs(data, nbTypesLabels, opti):
    """
    Find the Plackett-Luce coefficients of a ranking with the MM or MC algorithm.
    Return a dictionary with the coeffecients, the number of iterations
    and the execution time.
    """
    
    start_time = time.perf_counter()
    
    newdata = preparingData(data, nbTypesLabels)
    newdata = np.asarray(newdata).astype(int)
    
    #If no data in the end: impossible to predict the preferred. Say
    #All inviduals have the same chance to win.
    if newdata.size == 0:
        v = dict()
        v['strength'] = np.ones(nbTypesLabels)/nbTypesLabels
        v['iter'] = 0
        v['execTime'] = time.perf_counter() - start_time
        return v
    
    #If one dimension (only the winners): only a frequence.
    if newdata.shape[1] == 1:
        
        unique, counts = np.unique(newdata, return_counts=True)
        strength = np.zeros(nbTypesLabels)
        strength[unique-1] = counts
        
        v = dict()
        v['strength'] = strength/np.sum(counts)
        v['iter'] = 0
        v['execTime'] = time.perf_counter() - start_time
        return v 
    
    if opti == 'MM':
        #Parameters for the MM algorithm.
        param = mmParameters(newdata, nbTypesLabels)
        v = __mmAlgorithm__(newdata, nbTypesLabels, param['w'], param['delta'])
    elif opti == 'MC':
        param = markovParameters(newdata, nbTypesLabels)
        v = __markovAlgorithm__(newdata, nbTypesLabels, param['W'], param['L'], param['C'])
    else:
        raise ValueError('Unknown optimisation algorithm was given.')
    
    execTime = time.perf_counter() - start_time
    v['execTime'] = execTime
    
    return v

def preparingData(data, nbTypesLabels):
    """
    Prepare the data to be exploitable for the algorithms.
    """
    
    #If only missing information: we can't predict anything.
    if np.count_nonzero(data) == 0:
        return []
               
    #Remove instances where no information on labels is given (for stability).
    newdata = data[~np.all(data == 0, axis=1)]
    
    #Make the matrix suitable for computation: all zeros at the end of the lines.
    nbContests = newdata.shape[0] #Number of contests.
    maxPlaces = newdata.shape[1] #Maximal number of places per contest.
    for i in range(0, nbContests):

        line = newdata[i,:]
        lineNonZeros = line[line!=0]
        
        newdata[i,:] = np.zeros(maxPlaces)
        newdata[i,0:lineNonZeros.shape[0]] = lineNonZeros
    
    #Now we delete all columns full of zeros (useless)
    newdata = newdata.T[~np.all(newdata == 0, axis=0)].T
    
    #If all rows are the same: we add a reverse ranking to make
    #computation possible.
    if(np.all(newdata == newdata[0,:]) == True):
        newdata = np.vstack([newdata, np.flip(newdata[0,:])])
    
    return newdata

def mmParameters(data, nbTypesLabels):
    """
    Find the parameters necessary to compute the coefficients of a Plackett-Luce
    model for a ranking. Return the number of inviduals between all instances,
    the number of instances, the maximum number of ranking inviduals among
    all instances, the winner for each observation and delta which tell
    the position of each individual for all observations.
    """
    
    nbContests = data.shape[0] #Number of contests.
    maxPlaces = data.shape[1] #Maximal number of places per contest.
    w = np.zeros(nbTypesLabels) #Number of times not last in the contest.
    
    #w is the number of contests where individual ith is not the last.
    ranked = np.count_nonzero(data,axis=1)
    for i in range(0, nbContests):
        for j in range(0,ranked[i]-1):
            w[data[i,j]-1] += 1
       
    #Delta: place of invidual i in contest j.
    #If i didn't participate : 0.
    #Delta[i,j] = ranking of the individual i in the contest j + count
    #of all individuals ranked on the previous contests.
    delta = np.zeros((nbTypesLabels, nbContests)).astype(int)
    place = 1
    for i in range(0,nbContests): 
        for j in range(0,maxPlaces):
            if(data[i,j] > 0):
                delta[data[i,j]-1,i] = place
            place = place+1
            
    d = dict()
    d['w'] = w
    d['delta'] = delta
    return d

def __mmAlgorithm__(data, nbTypesLabels, w, delta, tol = 1e-6, itrMax = 1000):
    """
    Algorithm to find the MLE of the Plackett-Luce coefficients, with the
    MM algorithm. Return the coefficients and the number of iterations.
    """
    
    #Based on DAVID R. HUNTER algorithm in mathlab (Hunter, 2004).
    strength = np.ones(nbTypesLabels)/nbTypesLabels
    dstrength = np.ones(nbTypesLabels)
    k = 0
    
    nbContests = data.shape[0] #Number of contests.
    maxPlaces = data.shape[1] #Maximal number of places per contest.
    data = data.T
    
    #Matrix giving ones when the indiidual is not last in the constest.
    posNonLast= np.ones((maxPlaces,nbContests))
    for j in range(0, nbContests):
        nonZeros = np.nonzero(data[:,j])
        posNonLast[nonZeros[0][-1]:maxPlaces,j] = 0

        
    while np.linalg.norm(dstrength, 2) > tol and k < itrMax:
    
        #Sum of the strength for places i and higher in the jth contest.
        g = np.array(data)
        g = np.where(data > 0, strength[data-1], g)
        g = np.cumsum(np.flip(g[0:maxPlaces,:],0),0)
        g = np.flip(g, axis=0)
        
        #Last positions are zero. We use a mask to know what is the last
        #position for each contest.
        g = np.where(posNonLast == 0, 0, g)
        
        #Reciprocal (when possible, otherwise zero)
        g = np.divide(1, g, out=np.zeros_like(g), where=g!=0)
    
        #Sum of all denominators for ith place in jth contest.
        g = np.cumsum(g,0)
        
        #Sum of all denominators when delta is not zero.
        flatDelta = np.ndarray.flatten(delta)
        flattenG = np.ndarray.flatten(g, 'F')
        dem = np.where(flatDelta > 0, flattenG[flatDelta-1], 0)
        dem = np.reshape(dem,(nbTypesLabels, nbContests),'C')
        sumDem = np.sum(dem, 1)
        
        newstrength = np.divide(w, sumDem, out=np.zeros_like(sumDem), where=sumDem!=0)
        dstrength = newstrength - strength
        strength = newstrength
        
        k = k+1
    
    strength = strength/np.sum(strength)     
    d = dict()
    d['strength'] = strength
    d['iter'] = k
    return d

def markovParameters(data, nbTypesLabels, opt = 'partial'):
    """
    Find the parameters necessary to compute the coefficients of a Plackett-Luce
    model for a ranking. Return the number of inviduals between all instances,
    the number of instances, the maximum number of ranking inviduals among
    all instances, the winner for each observation, the losers for each
    observation and the participants for each observation.
    """
    
    if opt == 'partial':
        val = 1
    elif opt == 'full':
        val = 0
    else:
        raise ValueError('Unknown optimisation algorithm was given.')
        
    nbContests = data.shape[0] #Number of contests.
    maxPlaces = data.shape[1] #Maximal number of places per contest.

    #Per contest, there are maxPlaces - 1 observations. 
    #The first observation is among the maxPlaces elements. 
    #The second we don't consider the first element. 
    #The third we don't consider the first and second elements... 
    #Until the last observation of the ranking between the last two elements.
    nbObservations = nbContests * (maxPlaces-val) 
    #1 if ith is the first element of the jth observation.
    W = np.zeros((nbTypesLabels, nbObservations)).astype(int)
    #1 if ith is part of the jth observation but not the winner (loser).
    L = np.zeros((nbTypesLabels, nbObservations)).astype(int)
    #1 if ith is part of the jth observation.
    C = np.zeros((nbTypesLabels, nbObservations)).astype(int) 
    
    for i in range(0,nbContests):
        for j in range(0,maxPlaces-val):
            
            if data[i,j] == 0:
                continue
            
            row = i * (maxPlaces-val)
            
            winner = data[i,j]-1
            W[winner, row + j] = 1
            
            losers = np.trim_zeros(data[i,(j+1):maxPlaces])-1
            L[losers, row + j] = 1
        
    C = W+L
    
    d = dict()
    d['W'] = W
    d['L'] = L
    d['C'] = C
    return d


def __markovAlgorithm__(data, nbTypesLabels, W, L, C, tol = 1e-04, itrMax = 50):
    """
    Algorithm to find the MLE of the Plackett-Luce coefficients, with the
    markov algorithm. Return the coefficients and the number of iterations.
    """
    
    #Based on (Gu Yin, 2019)
    strength = np.ones(nbTypesLabels)/nbTypesLabels
    dstrength = np.ones(nbTypesLabels)
    k = 0
        
    nbContests = data.shape[0] #Number of contests.
    maxPlaces = data.shape[1] #Maximal number of places per contest.
    
    
    while np.linalg.norm(dstrength, 2) > tol:
        
        sigma = np.zeros((nbTypesLabels, nbTypesLabels))
        
        #Vectorised form of the algorithm presented in the article.
        #We calculate sigma, the transition matrix, for i != j.
        #The code is from the R code of the authors of the article :
        #https://github.com/PaparazziGG/Fast-Algorithm-for-Generalized-Multinomial-Models-with-Ranking-Data
        
        ones = np.ones(nbContests * (maxPlaces - 1))[:, np.newaxis]
        wstrength = np.dot(W.T, strength[:, np.newaxis])
        cstrength = np.dot(C.T, strength[:, np.newaxis])
        sigmaDivision = np.divide(np.divide(ones,wstrength),cstrength)
        secondDotTerm = np.multiply(sigmaDivision,L.T)
        
        vWin = np.multiply(strength[:, np.newaxis], W)
        sigma = np.dot(vWin,secondDotTerm).T
        
        #Calculating the diagonal of our matrix and normalise the matrix
        #to have a stochastic matrix, ie each row sum is 1 and each element
        #is a probability (between 0 and 1).
                    
        np.fill_diagonal(sigma, 1-np.sum(sigma,axis=1)) 
        np.fill_diagonal(sigma, np.diagonal(sigma)-np.min(np.diagonal(sigma)))
        sigma = sigma/np.sum(sigma)*nbTypesLabels
        
        #Proposition 2.14.1 of Resnick to have the stationnary distribution
        #of an irreductible stochastic matrix. 
        stationaryToInverse = np.eye(nbTypesLabels) - sigma + np.ones((nbTypesLabels,nbTypesLabels))
        inv = np.linalg.inv(stationaryToInverse)
        newstrength = np.dot(np.ones(nbTypesLabels),inv)
        
        dstrength = strength - newstrength
        strength = newstrength
        k = k+1
                
    d = dict()
    d['strength'] = strength
    d['iter'] = k
    return d
