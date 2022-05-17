from model.contour_pl import generateNormL, overAlpha #For contour likelihood method.
from model.abstention_pl import rankingAbstention, preferenceMatrix #For abstention method.
from model.opti_pl import coefs #For a plackett-luce model.

import numpy as np
from choix import generate_rankings #Function used to generate ranking from PL parameters.

#The parameters of the PL models. Each PL model should have the same number of paramters.
#The sum of each should be equal to one.
pl_parameters_1 = [0.05,0.5,0.2,0.15,0.1]
pl_parameters_2 = [0.25,0.15,0.5,0.05,0.05]
#Coefficients for the mixture. 0 if no mixture. Sum of the two must be equal to one.
coef_1 = 0.5
coef_2 = 0.5
#The number of rankings to be generated for the synthetic data set.
nb_rankings = 200
nb_rankings_1 = int(np.floor(nb_rankings * coef_1))
nb_rankings_2 = int(np.ceil(nb_rankings * coef_2))

nbPoints = 10000 #Number of points to generate for the contour function.
dirichCoefs = [10,100,1000] #Coeffecients for the Dirichlest distributions: alpha = nu * coef * dim
size_rankings = 5 #The number of labels in each rankings. It should never be over the number of elements in pl_parameters.
threshold = 0.5 #The threshold for the abstention method.
beta = 1 #The beta cut for the contour likelihood method.

#Generates the rankings from the parameters (pl parameters, number of rankings, size of rankings) previously set.
rankings_1 = np.asarray(generate_rankings(pl_parameters_1, nb_rankings_1, size_rankings)).astype(int)
rankings_2 = np.asarray(generate_rankings(pl_parameters_2, nb_rankings_2, size_rankings)).astype(int)
rankings = np.vstack((rankings_1, rankings_2))
rankings = rankings +1 #One is added, because the generation start at 0 and zeros would be considered as missing labels by the algorithms.

nbLabels = len(pl_parameters_1) #Number of different labels.

#Determination of the optimal nu parameters.
opti_nu = coefs(rankings, nbLabels, 'MM')['strength'] #Get the MLE estimate.

### Contour likelihood ###

#Generate 3*100 PL parameters according to three different Dirichlet distributions + MLE.
#Return the contour likelihood for each PL parameters, and the PL parameters.
l_norm, contour_nu = generateNormL(opti_nu, rankings, nbLabels, dirichCoefs, nbTraining = nbPoints)
nu_selected, l_selected = overAlpha(l_norm, contour_nu, beta)
#If beta = 1: check only one point is selected.
if beta == 1 and nu_selected.ndim > 1:
    nu_selected = nu_selected[0,:]
    l_selected = l_selected[0]

contour_pref = preferenceMatrix(nu_selected, nbLabels) #All preference matrices.
contour_order = rankingAbstention(contour_pref, threshold = 0.5) #All order matrices.

### Abstention method ###

abs_pref = preferenceMatrix(opti_nu, nbLabels) #Preference matrix.
abs_order = rankingAbstention(abs_pref, threshold) #Order matrix.

### Difference ###

#0 if identical. 
#1 if abstention method predicted an order but not contour likelihood method.
#-1 if contour likelihood method predicted an order, but not abstention method.
diff_order = abs_order - contour_order