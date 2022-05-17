import numpy as np

from model.opti_pl import markovParameters, coefs

class IBPlLearner:
    
    def __init__(self, neighbourhoodType, K, opti, nbTypesLabels):
        """
        Number of neighbours to consider and which algorithm to find the MLE
        of the Plackett-Luce (MM vs Markov (MC))
        """
        
        self.__K = K
        if neighbourhoodType == 'fixed' or neighbourhoodType == 'distance':
            self.__neighbourhoodType = neighbourhoodType
        else:
            raise ValueError("Not a known way to find neighbours.")
            
        if opti == 'MM' or opti == 'MC':
            self.__opti = opti
        else:
            raise ValueError("Not a known algorithm.")
            
        self.__nbTypesLabels = nbTypesLabels
        self.__W = 0
        self.__C = 0
        
    def loglikelihood(self, v, y = []):
        """
        Calculate the likelihood of a plackett-luce model from its coefficient,
        the win matrix of the data and the participant matrix.
        """
        
        if v.ndim == 1:
            v = v[None,:]
        nbRep = v.shape[0]
        
        if y == []:
            W = self.__W
            C = self.__C
        else:
            params = markovParameters(y, self.__nbTypesLabels)        
            W = params['W']
            C = params['C']
            #Remove lines and columns with zeros.
            W = W.T[~np.all(W == 0, axis=0)].T
            #W = W[~np.all(W == 0, axis=1)]
            C = C.T[~np.all(C == 0, axis=0)].T
            #C = C[~np.all(C == 0, axis=1)]
        
        fullW = np.repeat(W[np.newaxis,:,:], nbRep, axis = 0)
        fullC = np.repeat(C[np.newaxis,:,:], nbRep, axis = 0)
        fullV = np.repeat(v[:,:,np.newaxis], W.shape[1], axis = 2)
        
        vW = np.multiply(fullW, fullV)
        vC = np.multiply(fullC, fullV)
        
        vWSum = np.sum(vW, axis = 1)
        vCSum = np.sum(vC, axis = 1)
        
        l = np.sum(np.log(vWSum) - np.log(vCSum), axis = 1)
        
        return l
        
    def findClosestNeighbours(self, element):
        """
        Find the closest neighbours of an element after it was fitted.
        """
        
        neighbours = self.__Xtrain
        dist_2 = np.linalg.norm(neighbours - element, axis = 1)
        dist_ordered = np.sort(dist_2)
        ind_ordered = np.argsort(dist_2)
        
        #Take the one according to a distance.
        if self.__neighbourhoodType == 'distance':
            k_closest = ind_ordered[dist_ordered <= self.__K]
        #Percentile
        elif self.__neighbourhoodType == 'percentile':
            distThreshold = np.percentile(dist_ordered, self.__K) 
            k_closest = ind_ordered[dist_ordered <= distThreshold]
        #k nearest.
        elif self.__neighbourhoodType == 'fixed':
            k_closest = ind_ordered[0:self.__K]
        
        neighbours_ranking = self.__ytrain[k_closest,:].astype(int)

        return neighbours_ranking
            
    def fit(self, X, y):
        """
        Fit a model by taking the training data sets.
        """
        
        self.__Xtrain = X
        self.__ytrain = y
        
        params = markovParameters(y, self.__nbTypesLabels)        
        self.__W = params['W']
        self.__C = params['C']
        #Remove lines and columns with zeros.
        self.__W = self.__W.T[~np.all(self.__W == 0, axis=0)].T
        #W = W[~np.all(W == 0, axis=1)]
        self.__C = self.__C.T[~np.all(self.__C == 0, axis=0)].T
        #C = C[~np.all(C == 0, axis=1)]
        
        return self
    
    def predict(self, X):
        """
        Predict with the Instance-Based method: find the K closest neighbours
        and from them try to fit a Plackett-Luce model.
        """
        
        if X.ndim == 1:
            X = X[np.newaxis,...]
            
        nbRanking = X.shape[0]
        nbRanked = self.__ytrain.shape[1]
        strength = np.zeros((nbRanking, nbRanked))
        ranking = np.zeros((nbRanking, nbRanked)).astype(int)
        
        for i in range(0, nbRanking):
                   
            element = X[i,:]
            neighbours_ranking = self.findClosestNeighbours(element)
                
            #Plackett-Luce with k nearest.
            v = coefs(neighbours_ranking, self.__nbTypesLabels, self.__opti)
            
            strength[i,:] = v['strength']
            sortedstrength = np.flip(np.argsort(v['strength']))[0:nbRanked]
            ranking[i,0:nbRanked] = sortedstrength[0:nbRanked]+1
            
        return strength, ranking