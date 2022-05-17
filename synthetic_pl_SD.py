from model.opti_pl import coefs #For a plackett-luce model.
from model.contour_pl import generateNormL, overAlpha #For contour likelihood method.
from model.abstention_pl import preferenceMatrix, rankingAbstention #For abstention method.

import numpy as np
from choix import generate_rankings #Function used to generate ranking from PL parameters.

def generate_order_beta(beta,l_norm, contour_nu):
    nu_selected, l_selected = overAlpha(l_norm, contour_nu, beta)
    contour_pref = preferenceMatrix(nu_selected, nbLabels) 
    contour_order = rankingAbstention(contour_pref, threshold = 0.5)
    minmat=np.minimum(contour_order[0],contour_order[0])
    for y in contour_order:
        minmat=np.minimum(minmat,y)
    return minmat
    
def generate_list_order_lik(l_norm,contour_nu,size):
    mat_list=np.zeros((0,size,size))
    for val in np.arange(0.00,1.0,0.005):
        mat_list=np.append(mat_list,np.expand_dims(generate_order_beta(val,l_norm, contour_nu),axis=0),axis=0)
    return np.unique(mat_list,axis=0)

def generate_list_order_abs(abs_pref,size):
    mat_list=np.zeros((0,size,size))
    for val in np.arange(0.5,1.0,0.005):
        mat_list=np.append(mat_list,rankingAbstention(abs_pref, val),axis=0)
    return np.unique(mat_list,axis=0)

def compare_rankings(list_lik,list_abs):
    inequal=0
    if np.size(list_lik,axis=0)!=np.size(list_abs,axis=0):
        print("Oops, not same set of rankings")
    for i in range(np.size(list_lik,axis=0)):
        if not np.array_equal(abs_lik[i],abs_ord[i]):
            inequal=inequal+1
    return inequal

#The parameters of the PL models. Each PL model should have the same number of paramters.
#The sum of each should be equal to one.
pl_parameters_1 = [0.1,0.2,0.2,0.4,0.1]
pl_parameters_2 = [0.4,0.1,0.15,0.3,0.05]
#Coefficients for the mixture. 0 if no mixture. Sum of the two must be equal to one.
coef_1 = 0.7
coef_2 = 0.3
#The number of rankings to be generated for the synthetic data set.
nb_rankings = 5
nb_rankings_1 = int(np.floor(nb_rankings * coef_1))
nb_rankings_2 = int(np.ceil(nb_rankings * coef_2))

size_rankings = 5 #The number of labels in each rankings. It should never be over the number of elements in pl_parameters.
threshold = 0.5 #The threshold for the abstention method.
beta = 0.95 #The beta cut for the contour likelihood method.

#Generates the rankings from the parameters (pl parameters, number of rankings, size of rankings) previously set.
rankings_1 = np.asarray(generate_rankings(pl_parameters_1, nb_rankings_1, size_rankings)).astype(int)
rankings_2 = np.asarray(generate_rankings(pl_parameters_2, nb_rankings_2, size_rankings)).astype(int)
rankings = np.vstack((rankings_1, rankings_2))
rankings = rankings +1 #One is added, because the generation start at 0 and zeros would be considered as missing labels by the algorithms.

nbLabels = len(pl_parameters_1) #Number of different labels.

### Contour likelihood ###

#Generate 3*100 PL parameters according to three different Dirichlet distributions + MLE.
#Return the contour likelihood for each PL parameters, and the PL parameters.
l_norm, contour_nu = generateNormL(rankings, nbLabels, nbTraining = 50000)
nu_selected, l_selected = overAlpha(l_norm, contour_nu, beta)

contour_pref = preferenceMatrix(nu_selected, nbLabels) #All preference matrices.
contour_order = rankingAbstention(contour_pref, threshold = 0.5) #All order matrices.

### Abstention method ###

abs=[]
abs_nu = coefs(rankings, nbLabels, 'MM')['strength'] #Get the MLE estimate. Should be the same as contour_nu[0].
abs_pref = preferenceMatrix(abs_nu, nbLabels) #Preference matrix.
abs_order = rankingAbstention(abs_pref, threshold) #Order matrix.

### Difference ###

#0 if identical. 
#1 if abstention method predicted an order but not contour likelihood method.
#-1 if contour likelihood method predicted an order, but not abstention method.
diff_order = abs_order - contour_order

abs_lik=generate_list_order_lik(l_norm,contour_nu,size_rankings)
abs_ord=generate_list_order_abs(abs_pref,size_rankings)

print('number of disagreeing orders')
print(compare_rankings(abs_lik,abs_ord))



