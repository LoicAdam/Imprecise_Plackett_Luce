import data.data_ranking as data_ranking
from model.abstention_pl import preferenceMatrix, rankingAbstention
from model.contour_pl import impreciseOrder
from model.evaluation_pl import completeness, correctness, rankingToOrderMatrix
from model.tools_pl import algoChoice, perturbMatrix

import tikzplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

def dataEvolution(X, y, training_kept, nbR = 1, nbF = 10, seed = 0, p_type = 'miss', p = 0, 
                  algo = 'IB', opti = 'MM', 
                  alpha = [1], threshold = [0.5], nbTraining = 200, dCoefs = [1]):
    
    #Pertub the rankings
    y_pert = perturbMatrix(y, p_type, p)
            
    nbIndividuals = np.max(y)
    numberPredicted = np.zeros((len(training_kept)))
    meanCpImp = np.zeros((len(training_kept)))
    meanCpAbst = np.zeros((len(training_kept)))
    varCpImp = np.zeros((len(training_kept)))
    varCpAbst = np.zeros((len(training_kept)))
    meanCrImp = np.zeros((len(training_kept)))
    meanCrAbst = np.zeros((len(training_kept)))
    varCrImp = np.zeros((len(training_kept)))
    varCrAbst = np.zeros((len(training_kept)))
    
    dist = []
    #The median distance
    for i in range(0, X.shape[0]-1):
        
        element = X[i,:]
        others = X[i+1:,:]
        dist_2 = np.linalg.norm(others - element, axis = 1)
        dist = np.hstack((dist, dist_2))
        
    K = np.percentile(dist, 50) 
    
    kf = RepeatedKFold(n_splits= nbF, n_repeats = nbR, random_state = seed)
    
    for train_index, test_index in kf.split(X):
                
        X_train_ori, X_test = X[train_index,:], X[test_index,:]
        y_train_ori, y_test = y_pert[train_index,:], y[test_index,:]
    
        for t in range(0, len(training_kept)):
            
            #Get datasets
            if training_kept[t] == 1:
                X_train = X_train_ori
                y_train = y_train_ori
            else:
                indices = np.random.permutation(X_train_ori.shape[0])
                indexSplit = int(np.floor(training_kept[t] * len(indices)))
                training_idx = indices[:indexSplit]
                X_train = X_train_ori[training_idx,:]
                y_train = y_train_ori[training_idx,:]
                
            scaler = StandardScaler()
            scaler.fit(X_train)
            scaler.transform(X_train)
            scaler.transform(X_test)
            
            #Likelihood
            imp_res = impreciseOrder(X_train, y_train, X_test, nbIndividuals, 
                                           alpha, nbTraining, algo, 'distance', 
                                           K, opti, dCoefs)
                
            imp_pref = imp_res[0][:,0,:,:]
            imp_pref = np.where(imp_pref < 1, 0, 1)
            imp_pref_ori = rankingToOrderMatrix(y_test, nbIndividuals)
            imp_cp = completeness(imp_pref, imp_pref_ori)
            imp_cr = correctness(imp_pref, imp_pref_ori)
            
            meanCpImp[t] = meanCpImp[t] + np.mean(imp_cp, axis = 0)
            varCpImp[t] = varCpImp[t] + np.var(imp_cp, axis = 0)
            meanCrImp[t] = meanCrImp[t] + np.mean(imp_cr, axis = 0)
            varCrImp[t] = varCrImp[t] + np.var(imp_cr, axis = 0)
            
            #Abstention
            tol = 10e-15
            modelpl = algoChoice(algo, nbIndividuals, 'distance', K, opti)
            modelpl.fit(X_train, y_train)
            gamma_predicted = modelpl.predict(X_test)[0]
            gamma_predicted = np.where(gamma_predicted < tol, tol, gamma_predicted)
            gamma_predicted = gamma_predicted/np.sum(gamma_predicted, axis=1)[:,None]
            
            abst_pref = preferenceMatrix(gamma_predicted, nbIndividuals)
            abst_order = rankingAbstention(abst_pref, threshold[0])
            abst_pref_ori = rankingToOrderMatrix(y_test, nbIndividuals)
            abst_cp = completeness(abst_order, abst_pref_ori)
            abst_cr = correctness(abst_order, abst_pref_ori)
              
            meanCpAbst[t] = meanCpAbst[t] + np.mean(abst_cp, axis = 0)
            varCpAbst[t] = varCpAbst[t] + np.var(abst_cp, axis = 0)
            meanCrAbst[t] = meanCrAbst[t] + np.mean(abst_cr, axis = 0)
            varCrAbst[t] = varCrAbst[t] + np.var(abst_cr, axis = 0)
            
            numberPredicted[t] = numberPredicted[t] + y_test.shape[1]
            
            print(t)
           
    cpImp = np.zeros((len(training_kept), 3))
    cpAbst = np.zeros((len(training_kept), 3))
    crImp = np.zeros((len(training_kept), 3))
    crAbst = np.zeros((len(training_kept), 3))
    
    meanCpAbst = np.divide(meanCpAbst, nbF*nbR)
    meanCpImp = np.divide(meanCpImp, nbF*nbR)
    varCpAbst = np.divide(varCpAbst, nbF*nbR)
    varCpImp = np.divide(varCpImp, nbF*nbR)
    meanCrAbst = np.divide(meanCrAbst, nbF*nbR)
    meanCrImp = np.divide(meanCrImp, nbF*nbR)
    varCrAbst = np.divide(varCrAbst, nbF*nbR)
    varCrImp = np.divide(varCrImp, nbF*nbR)
    
    sdCpImp = np.sqrt(varCpImp)
    cpImp[:, 0] = meanCpImp - 1.96 * np.divide(sdCpImp, np.sqrt(numberPredicted))
    cpImp[:, 1] = meanCpImp
    cpImp[:, 2] = meanCpImp + 1.96 * np.divide(sdCpImp, np.sqrt(numberPredicted))
            
    sdCpAbst  = np.sqrt(varCpAbst)
    cpAbst[:, 0] = meanCpAbst - 1.96 * np.divide(sdCpAbst, np.sqrt(numberPredicted))
    cpAbst[:, 1] =  meanCpAbst
    cpAbst[:, 2] = meanCpAbst + 1.96 * np.divide(sdCpAbst, np.sqrt(numberPredicted))
    
    sdCrImp = np.sqrt(varCrImp)
    crImp[:, 0] = meanCrImp - 1.96 * np.divide(sdCrImp, np.sqrt(numberPredicted))
    crImp[:, 1] = meanCrImp
    crImp[:, 2] = meanCrImp + 1.96 * np.divide(sdCrImp, np.sqrt(numberPredicted))
            
    sdCrAbst  = np.sqrt(varCrAbst)
    crAbst[:, 0] = meanCrAbst - 1.96 * np.divide(sdCrAbst, np.sqrt(numberPredicted))
    crAbst[:, 1] =  meanCrAbst
    crAbst[:, 2] = meanCrAbst + 1.96 * np.divide(sdCrAbst, np.sqrt(numberPredicted))
            
    return cpImp, cpAbst, crImp, crAbst

def plotCpEvolution(training_kept, cpImp, cpAbst):
    
    colors = [plt.cm.viridis(i) for i in np.linspace(0,0.8,2)]
    fig, ax = plt.subplots(1,1)

    training_removed = np.ones(len(training_kept)) - training_kept
    ax.plot(training_removed, cpImp[:,1], '-o', color = colors[0], label="Likelihood approach")
    ax.plot(training_removed, cpImp[:,0], '--', color = colors[0])
    ax.plot(training_removed, cpImp[:,2], '--', color = colors[0])
    ax.plot(training_removed, cpAbst[:,1], '-v', color = colors[1], label="Classic abstention")
    ax.plot(training_removed, cpAbst[:,0], '--', color = colors[1])
    ax.plot(training_removed, cpAbst[:,2], '--', color = colors[1])
    ax.set_xlabel('Training set removed')
    ax.set_ylabel('Completeness')
    ax.legend(loc="lower left")
    ax.xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    plt.savefig(data[2] + '_' + 'completeness.png', dpi=300)
    tikzplotlib.save(data[2] + '_' + 'completeness.tex')
    
def plotCrEvolution(training_kept, crImp, crAbst):
    
    colors = [plt.cm.viridis(i) for i in np.linspace(0,0.8,2)]
    fig, ax = plt.subplots(1,1)

    training_removed = np.ones(len(training_kept)) - training_kept
    ax.plot(training_removed, crImp[:,1], '-o', color = colors[0], label="Likelihood approach")
    ax.plot(training_removed, crImp[:,0], '--', color = colors[0])
    ax.plot(training_removed, crImp[:,2], '--', color = colors[0])
    ax.plot(training_removed, crAbst[:,1], '-v', color = colors[1], label="Classic abstention")
    ax.plot(training_removed, crAbst[:,0], '--', color = colors[1])
    ax.plot(training_removed, crAbst[:,2], '--', color = colors[1])
    ax.set_xlabel('Training set removed')
    ax.set_ylabel('Correctness')
    ax.legend(loc="lower left")
    ax.xaxis.set_ticks(np.arange(0, 1.2, 0.2))
    plt.savefig(data[2] + '_' + 'correctness.png', dpi=300)
    tikzplotlib.save(data[2] + '_' + 'correctness.tex')
           
if __name__ == '__main__':
    data = data_ranking.getAuthorship()
    training_size = [1,0.9,0.8,0.6,0.4,0.2,0.1]
    t = [0.54]
    a = [0.1]
    seed = 10
    
    cpImp, cpAbst, crImp, crAbst = dataEvolution(data[0], data[1], training_size, nbR = 5, nbF = 10, seed = seed,
                                  p_type = 'miss', p = 0, algo = 'IB', 
                                  opti = 'MM', alpha = a, threshold = t,
                                  nbTraining = 100, dCoefs = np.logspace(0,10,11))
    plotCpEvolution(training_size, cpImp, cpAbst)
    plotCrEvolution(training_size, crImp, crAbst)