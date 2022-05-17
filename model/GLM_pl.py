from model.opti_pl import markovParameters

import numpy as np
from scipy.optimize import minimize

class GLMPlLearner:
    
    def __init__(self, nbTypesLabels):
        
        self.__M = nbTypesLabels #Number of different labels.
        self.__D = 0 #Number of features for the instance space.
        
        self.__fullX = 0 #Instances adapted for log.
        self.__fullW = 0 #Winners matrix.
        self.__fullC = 0 #Contesters matrix.
        self.__X = 0 #Instances.
        self.__y = 0 #Rankings.
        
        self.__alpha = 0 #GLM coefficients
    
    def __v__(self, alpha, x):
        
        alphaX = np.multiply(alpha, x)
        vSum = np.sum(alphaX, axis = 0)
        v = np.where(vSum != 0, np.exp(vSum), 0)
        return v
    
    '''def __delta__(self, yinv, m):
        return np.where(yinv >= m, 1, 0)
    
    def __jacobian__(self, X):
                
        #Inverse of the rankings. Used for the computation of the deltas.
        y = self.__y
        values = y[y > 0]
        indices = np.argwhere(y > 0)[:,0]
        yinv = np.zeros((self.__y.shape[0], self.__y.shape[1]))
        yinv[indices,values-1] = values
        
        #First part of the equation.
        ones = np.where(yinv > 0, 1, 0)
        fullOnes = np.repeat(ones[:,:,np.newaxis], self.__D, axis = 2)
        fullX = np.repeat(X[:,np.newaxis,:], self.__M, axis = 1)
        
        firstSum = np.sum(np.multiply(fullOnes, fullX), axis = 0)
        
        #Second part of the equation.
        alpha = self.__alpha.reshape((self.__M, self.__D)).T
        
        ##Delta
        m = np.repeat(np.arange(0, self.__M)[np.newaxis, :], self.__M, axis = 0)
        m = np.repeat(m[np.newaxis, :, :], self.__y.shape[0] , axis = 0)
        yinvCoefs = np.repeat(yinv[:,:,np.newaxis], self.__M, axis = 2)
        coefs = self.__delta__(yinvCoefs, m+1)
        coefs = np.repeat(coefs[:, :, :, np.newaxis], self.__D , axis = 3)
        
        ##Numerator
        va = np.repeat(alpha[:,:,np.newaxis], self.__y.shape[0], axis = 2)
        vn = np.repeat(self.__X.T[:, np.newaxis, :], self.__M, axis = 1)
        vnum = self.__v__(va, vn)
        vnum = np.repeat(vnum[np.newaxis,:,:], self.__D, axis = 0)
        num = np.multiply(vnum, vn)
        num = np.repeat(num.T[:,np.newaxis,:,:], self.__M, axis = 1)
        
        
        #Denominator
        pl_opti = OptimisationPL()
        param = pl_opti.markovParameters(y, self.__M, 'full')
        C = param['C']
        fullC = C.reshape((self.__M, y.shape[0], self.__M))
        fullC = np.transpose(fullC, (1,2,0))
        fullC = np.repeat(fullC[:,:,:,np.newaxis], self.__D, axis = 3)
        fullA = np.repeat(alpha.T[np.newaxis, :, :], self.__M, axis = 0)
        fullA = np.repeat(fullA[np.newaxis, :, :, :], self.__y.shape[0], axis = 0)
        fullAC = np.where(fullC != 0, fullA, 0)
        fullXdem = np.repeat(fullX[:,np.newaxis,:,:], self.__M, axis = 1)
        vdem = self.__v__(fullAC.T, fullXdem.T)
        dem = np.sum(vdem, axis = 1).T
        dem = np.repeat(dem[:,np.newaxis,], self.__M, axis = 1)
        dem = np.repeat(dem[:,:,:,np.newaxis], self.__D, axis = 3)
        
        with np.errstate(divide = 'ignore'):
            division = np.where(dem > 0, np.divide(num, dem), 0)
        divisionCoefs = np.multiply(coefs, division)
        
        secondSum = np.sum(divisionCoefs, axis = 1)
        secondSum = np.sum(secondSum, axis = 0)
        
        jac = firstSum - secondSum
        return -np.ravel(jac)'''
                
    def loglikelihood(self, X = [], y = [], flagOpti = False):
        '''
        Loglikelihood according the GLM procedure
        '''
        
        if flagOpti == False:
            
            param = markovParameters(y, self.__M) 
            
            #Operations to do an optimise computation of the loglikelihood.
            W = param['W']
            C = param['C']
            fullW = np.repeat(W[np.newaxis,:,:], self.__D, axis = 0)
            fullC = np.repeat(C[np.newaxis,:,:], self.__D, axis = 0)
            fullX = np.repeat(X, repeats = (np.ones(X.shape[0])*(self.__M-1)).astype(int), axis = 0).T
            fullX = np.repeat(fullX[:,np.newaxis,:], self.__M, axis = 1)
            
        else:
            
            fullW = self.__fullW
            fullC = self.__fullC
            fullX = self.__fullX
         
        alpha = self.__alpha.reshape((self.__M, self.__D)).T
        fullA = np.repeat(alpha[:,:,np.newaxis], fullW.shape[2], axis = 2)
        fullAW = np.where(fullW != 0, fullA, 0)
        fullAC = np.where(fullC != 0, fullA, 0)
    
        #No need to do an exponential, since log(exp(x)) = x
        mulAW = np.multiply(fullAW, fullX)
        addAW = np.sum(mulAW, axis = 0)
        vW = np.sum(addAW, axis = 0)
        
        vC = self.__v__(fullAC, fullX)
        sumC = np.sum(vC, axis = 0)      
        
        #No need to worry about divide by zeros, they're taken care of.
        with np.errstate(divide = 'ignore'):
            logSumC = np.where(sumC > 0, np.log(sumC), 0)
           
        log = vW - logSumC
        if flagOpti == False:
            log = np.reshape(log, ((X.shape[0], self.__M-1)))
            log = np.sum(log, axis = 1)
            return log
        else:
            log = np.sum(log)
            return log
    
    #Actually the negative of the loglikelihood (for optimisation).
    #Alpha has to be flatten (again because of optimisation method. Hur).
    def __optiLikelihood__(self, alpha):
        '''
        The opposite of the loglikelihood, so it can be optimised by scipy.
        '''
        self.__alpha = alpha
        optiL = -self.loglikelihood(flagOpti = True)
        return optiL
    
    '''def __optiJacobian__(self, alpha):
        self.__alpha = alpha
        return -self.__jacobian__(self.__X)'''
        
    def fit(self, X, y):
            
        self.__X = X
        self.__y = y
        self.__D = X.shape[1]
        
        #Get the matrices of winners and losers.
        #With those, it is possible to calculate the loglikelihood.
        param = markovParameters(y, self.__M) 
        
        #Operations to do an optimise computation of the loglikelihood.
        W = param['W']
        C = param['C']
        self.__fullW = np.repeat(W[np.newaxis,:,:], self.__D, axis = 0)
        self.__fullC = np.repeat(C[np.newaxis,:,:], self.__D, axis = 0)
        self.__alpha = np.ravel(np.random.rand(self.__M, self.__D))
        fullX = np.repeat(X, repeats = (np.ones(X.shape[0])*(self.__M-1)).astype(int), axis = 0).T
        self.__fullX = np.repeat(fullX[:,np.newaxis,:], self.__M, axis = 1)

        #Optimisation algorithm.
        res = minimize(self.__optiLikelihood__, self.__alpha, 
                       #jac = self.__optiJacobian__, 
                       method = 'L-BFGS-B',
                       options = {'disp':True})
        self.__alpha = (res.x).reshape((self.__M, self.__D))
        
        return self
    
    def predict(self, X):
        
        N = X.shape[0]
        X = X.T
        alpha = self.__alpha.T
        fullA = np.repeat(alpha[:,:,np.newaxis], repeats = N, axis = 2)
        fullX = np.repeat(X[:,np.newaxis,:], repeats = self.__M, axis = 1)
        
        coefs = np.zeros((N, self.__M))
        rankings = np.zeros((N, self.__M))
        
        coefs = self.__v__(fullA, fullX)
        coefs = np.divide(coefs, np.repeat(np.sum(coefs, axis = 0)[np.newaxis,:], 
                                           repeats = self.__M, axis = 0))
        
        rankings = np.argsort(coefs, axis = 0)[::-1].T
        
        return coefs.T, rankings+1