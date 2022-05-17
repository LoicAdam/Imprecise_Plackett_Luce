from model.opti_pl import coefs, markovParameters
from inference import likelihood
from model.contour_pl import overAlpha

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tikzplotlib

neighbours = np.asarray([[2,1,3],[2,1,3],[2,1,3],[1,3,2],[1,3,2]])
nbTypesLabels = 3
cut_1 = 0.9
cut_2 = 0.5
dirichCoef = 5, 
nbTraining = 20000

#Get the MLE of the neighbourhood
v_opt = coefs(neighbours, nbTypesLabels, 'MM')['strength']

#Generate a number of strenghs according to a dirichlet distribution.
v_generated = np.random.dirichlet(alpha = dirichCoef * v_opt, size = nbTraining)
v = np.vstack((v_opt,v_generated))

#Get their loglikelihood
params = markovParameters(neighbours, nbTypesLabels)        
W = params['W']
C = params['C']
W = W.T[~np.all(W == 0, axis=0)].T
C = C.T[~np.all(C == 0, axis=0)].T
l = likelihood(neighbours, nbTypesLabels, v)

#Get the contour likelihood
l_contour = np.exp(l - l[0])

x = np.linspace(0, 1, 1000)
x_0 = np.linspace(0, 0.5, 1000)

##Plot all the points##
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(1-x, x, linestyle='-', c = 'k') #Triangle part
ax.set_xlabel('v1')
ax.set_ylabel('v2')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
plt.plot(x_0, x_0, linestyle='-', c = 'r', linewidth=3) # Seperation part
plt.plot(-x_0+0.5, x, linestyle='-', c = 'g', linewidth=3)
plt.plot(x, -x_0+0.5, linestyle='-', c = 'b', linewidth=3)
scat = ax.scatter(v[:,0],v[:,1], c = l_contour, s = 1)
plt.colorbar(mappable = scat)
plt.text(0.52, 0.52, 'v1=v2', c = 'r', fontsize=10)
plt.text(0.1, 0.92, 'v1=v3', c = 'g', fontsize=10)
plt.text(0.88, 0.15, 'v2=v3', c = 'b', fontsize=10)
plt.savefig('full_contour.png', dpi=300)
tikzplotlib.save('full_contour.tex', encoding='utf-8')
    
##3D##
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(v[:,0],v[:,1],v[:,2], c = l_contour, s = 1)
ax.set_xlabel('v1')
ax.set_ylabel('v2')
ax.set_zlabel('v3')
ax.view_init(45,45)
plt.draw()
plt.savefig('contour_full_3d.png', dpi=300)
tikzplotlib.save('contour_full_3d.tex', encoding='utf-8')

##Plot all the points##
v_sel_1, l_sel_1 = overAlpha(l_contour, v, cut_1)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(1-x, x, linestyle='-', c = 'k') #Triangle part
ax.set_xlabel('v1')
ax.set_ylabel('v2')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
plt.plot(x_0, x_0, linestyle='-', c = 'r', linewidth=3) # Seperation part
plt.plot(-x_0+0.5, x, linestyle='-', c = 'g', linewidth=3)
plt.plot(x, -x_0+0.5, linestyle='-', c = 'b', linewidth=3)
scat = ax.scatter(v_sel_1[:,0],v_sel_1[:,1], c = l_sel_1, s = 1, vmin=0, vmax=1)
plt.colorbar(mappable = scat)
plt.text(0.52, 0.52, 'v1=v2', c = 'r', fontsize=10)
plt.text(0.1, 0.92, 'v1=v3', c = 'g', fontsize=10)
plt.text(0.88, 0.15, 'v2=v3', c = 'b', fontsize=10)
plt.savefig('contour_' + str(cut_1) + '.png', dpi=300)
tikzplotlib.save('contour_' + str(cut_1) + '.tex', encoding='utf-8')

##Plot all the points##
v_sel_2, l_sel_2 = overAlpha(l_contour, v, cut_2)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(1-x, x, linestyle='-', c = 'k') #Triangle part
ax.set_xlabel('v1')
ax.set_ylabel('v2')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
plt.plot(x_0, x_0, linestyle='-', c = 'r', linewidth=3) # Seperation part
plt.plot(-x_0+0.5, x, linestyle='-', c = 'g', linewidth=3)
plt.plot(x, -x_0+0.5, linestyle='-', c = 'b', linewidth=3)
scat = ax.scatter(v_sel_2[:,0],v_sel_2[:,1], c = l_sel_2, s = 1, vmin=0, vmax=1)
plt.colorbar(mappable = scat)
plt.text(0.52, 0.52, 'v1=v2', c = 'r', fontsize=10)
plt.text(0.1, 0.92, 'v1=v3', c = 'g', fontsize=10)
plt.text(0.88, 0.15, 'v2=v3', c = 'b', fontsize=10)
plt.savefig('contour_' + str(cut_2) + '.png', dpi=300)
tikzplotlib.save('contour_' + str(cut_2) + '.tex', encoding='utf-8')
