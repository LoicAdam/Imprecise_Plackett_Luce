from model.contour_pl import impreciseOrder
from model.abstention_pl import preferenceMatrix, rankingAbstention
from model.evaluation_pl import kendalltau, correctness, completeness, rankingToOrderMatrix
from model.tools_pl import algoChoice, perturbMatrix

import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

def modelPredict(X_train, y_train, X_test, y_test = [], p_type = 'miss', p = 0,
                 algo = 'IB', neighbourhoodType = 'fixed', K=15, opti='MM'):
    """
    Prediction of rankings.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)
                
    nbTypesLabels = np.max(y_train)
    modelpl = algoChoice(algo, nbTypesLabels, neighbourhoodType, K, opti)
    
    y_train = perturbMatrix(y_train, p_type, p)
                    
    modelpl.fit(X_train, y_train)
    model_predict = modelpl.predict(X_test)[1]
    d = dict()
    d['y'] = model_predict
    
    if y_test.size != 0:
        
        distAll = kendalltau(model_predict,y_test)
        distAvg = np.mean(distAll)
        d['dist'] = distAvg
        
    return d

def __crossValidationForModel__(trainingMethod, params, X, y, y_pert = [], 
                                training_size = 0.8, nbF = 10, nbR = 5, algo = 'IB', 
                                neighbourhoodType = 'fixed', K = 10, opti = 'MM', seed = 0):
    """
    General method to do cross validation with K-fold or simply with
    training/test sets.
    In the parameters, a method and its parameters have to be passed to tell
    what metrics the method has to compute.
    y_pert is the set of pertubated rankings (with missing labels or switched
    labels for example).
    """
    
    #Check if some pertubation is done on the rankings.
    if y_pert == []:
        y_pert = y
        
    nbTypesLabels = np.max(y)
    res = []
    
    #K folds
    if nbF > 1:
        
        kf = RepeatedKFold(n_splits= nbF, n_repeats = nbR, random_state = seed)
        fold = 0
        
        for train_index, test_index in kf.split(X):
            
            X_train, X_test = X[train_index,:], X[test_index,:]
            y_train, y_test = y_pert[train_index,:], y[test_index,:]
            
            scaler = StandardScaler()
            scaler.fit(X_train)
            scaler.transform(X_train)
            scaler.transform(X_test)
            
            tmpRes = trainingMethod(X_train, X_test, y_train, y_test,
                                    algo, nbTypesLabels, neighbourhoodType, 
                                    K, opti, *params)
            
            if res == []:
                res = tmpRes
            else:
                res = np.hstack((res, tmpRes))
            
            fold+= 1
            print(fold)
        
        #Training/Test
        else:
            
            indices = np.random.permutation(X.shape[0])
            indexSplit = int(np.floor(training_size * len(indices)))
            training_idx, test_idx = indices[:indexSplit], indices[indexSplit:]
            X_train, X_test = X[training_idx,:], X[test_idx,:]
            y_train, y_test = y_pert[training_idx,:], y[test_idx,:]
            
            scaler = StandardScaler()
            scaler.fit(X_train)
            scaler.transform(X_train)
            scaler.transform(X_test)
             
            tmpRes = trainingMethod(X_train, X_test, y_train, y_test,
                                    algo, nbTypesLabels, neighbourhoodType, 
                                    K, opti, *params)
            if res == []:
                res = tmpRes
            else:
                res = np.hstack((res, tmpRes))
    
    return res

def modelDistance(X, y, y_pert = [], training_size = 0.8, nbFolds = 10, nbRepeats = 5, algo = 'IB',
                  neighbourhoodType = 'fixed', K = 10, opti='MM'):
    """
    Cross validation with the instance based algorithm to compute the Kendall tau
    distance.
    """
    
    def __trainingMethod__(X_train, X_test, y_train, y_test, algo, nbTypesLabels,
                           neighbourhoodType, K, opti):
        
        modelpl = algoChoice(algo, nbTypesLabels, neighbourhoodType, K, opti)
        modelpl.fit(X_train, y_train)
        model_predict = modelpl.predict(X_test)[1]
        
        distAll = kendalltau(model_predict,y_test)
        return distAll
    
    res = __crossValidationForModel__(__trainingMethod__, [], 
                                      X, y, y_pert, training_size, nbFolds, nbRepeats, algo, 
                                      neighbourhoodType, K, opti)
    d = dict()
    d['Dist_Avg'] = np.mean(res)
    d['Dist_Std'] = np.std(res)
    d['Dist_Lower'] = d['Dist_Avg'] - 1.96 * np.divide(d['Dist_Std'], np.sqrt(len(res)))
    d['Dist_Upper'] = d['Dist_Avg'] + 1.96 * np.divide(d['Dist_Std'], np.sqrt(len(res)))
    return d

def modelImprecise(X, y, y_pert = [], training_size = 0.8, nbFolds = 10, nbRepeats = 5, 
                   algo = 'IB', neighbourhoodType = 'fixed', K = 10, opti ='MM', 
                   alpha = [0.95,0.9,0.8,0.7,0.6,0.5], nbTraining = 10000,
                   dirichCoefs = [1]):
    '''Cross validation with contour likelihood algorithm.'''
    
    def __trainingMethod__(X_train, X_test, y_train, y_test, algo, nbTypesLabels,
                           neighbourhoodType, K, opti, alpha, nbTraining, dirichCoefs):
        
        cp = np.zeros((y_test.shape[0], len(alpha)))
        cr = np.zeros((y_test.shape[0], len(alpha)))
    
        res = impreciseOrder(X_train, y_train, X_test, nbTypesLabels, 
                                       alpha, nbTraining, algo, neighbourhoodType, 
                                       K, opti, dirichCoefs)
            
        nba = 0
        for a in alpha:
            pref = res[0][:,nba,:,:]
            pref = np.where(pref < 1, 0, 1)
            pref_ori = rankingToOrderMatrix(y_test, nbTypesLabels)
            cp[:,nba] = completeness(pref, pref_ori)
            cr[:,nba] = correctness(pref, pref_ori)
            nba = nba+1
            
        return np.stack((cp,cr))
    
    res = __crossValidationForModel__(__trainingMethod__, (alpha, nbTraining, dirichCoefs), 
                                      X, y, y_pert, training_size, nbFolds, nbRepeats, algo, 
                                      neighbourhoodType, K, opti)
    
    d = dict()
    d['Completeness_Avg'] = np.mean(res[0,:,:], axis = 0)
    d['Completeness_Std'] = np.std(res[0,:,:], axis = 0)
    d['Completeness_Lower'] = d['Completeness_Avg'] - 1.96 * np.divide(d['Completeness_Std'], np.sqrt(res.shape[1]))
    d['Completeness_Upper'] = d['Completeness_Avg'] + 1.96 * np.divide(d['Completeness_Std'], np.sqrt(res.shape[1]))
    d['Correctness_Avg'] = np.mean(res[1,:,:], axis = 0)
    d['Correctness_Std'] = np.std(res[1,:,:], axis = 0)
    d['Correctness_Lower'] = d['Correctness_Avg'] - 1.96 * np.divide(d['Correctness_Std'], np.sqrt(res.shape[1]))
    d['Correctness_Upper'] = d['Correctness_Avg'] + 1.96 * np.divide(d['Correctness_Std'], np.sqrt(res.shape[1]))
    return d

def modelAbstention(X, y, y_pert = [], training_size = 0.8, nbFolds = 10, nbRepeats = 5, 
                    algo = 'IB', neighbourhoodType = 'fixed', K = 10, opti ='MM', 
                    threshold = [0.5,0.6,0.7,0.8,0.9,1]):
    '''Cross validation with algorithm based on abstention.'''
    
    def __trainingMethod__(X_train, X_test, y_train, y_test, algo, nbTypesLabels,
                           neighbourhoodType, K, opti, threshold):
        
        tol = 10e-15
        cp = np.zeros((y_test.shape[0], len(threshold)))
        cr = np.zeros((y_test.shape[0], len(threshold)))
    
        modelpl = algoChoice(algo, nbTypesLabels, neighbourhoodType, K, opti)
        modelpl.fit(X_train, y_train)
        strength_predicted = modelpl.predict(X_test)[0]
        strength_predicted = np.where(strength_predicted < tol, tol, strength_predicted)
        strength_predicted = strength_predicted/np.sum(strength_predicted, axis=1)[:,None]
                    
        nba = 0
        for t in threshold:
            pref = preferenceMatrix(strength_predicted, nbTypesLabels)
            order = rankingAbstention(pref, t)
            pref_ori = rankingToOrderMatrix(y_test, nbTypesLabels)
            cp[:,nba] = completeness(order, pref_ori)
            cr[:,nba] = correctness(order, pref_ori)
            nba = nba+1
            
        return np.stack((cp,cr))
    
    res = __crossValidationForModel__(__trainingMethod__, (threshold,), 
                                      X, y, y_pert, training_size, nbFolds, nbRepeats, algo, 
                                      neighbourhoodType, K, opti)
    d = dict()
    d['Completeness_Avg'] = np.mean(res[0,:,:], axis = 0)
    d['Completeness_Std'] = np.std(res[0,:,:], axis = 0)
    d['Completeness_Lower'] = d['Completeness_Avg'] - 1.96 * np.divide(d['Completeness_Std'], np.sqrt(res.shape[1]))
    d['Completeness_Upper'] = d['Completeness_Avg'] + 1.96 * np.divide(d['Completeness_Std'], np.sqrt(res.shape[1]))
    d['Correctness_Avg'] = np.mean(res[1,:,:], axis = 0)
    d['Correctness_Std'] = np.std(res[1,:,:], axis = 0)
    d['Correctness_Lower'] = d['Correctness_Avg'] - 1.96 * np.divide(d['Correctness_Std'], np.sqrt(res.shape[1]))
    d['Correctness_Upper'] = d['Correctness_Avg'] + 1.96 * np.divide(d['Correctness_Std'], np.sqrt(res.shape[1]))
    return d