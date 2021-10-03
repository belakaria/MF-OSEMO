# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:57:25 2019

@author: Syrine Belakaria
This code is partly based on the code from https://github.com/takeno1995/BayesianOptimization
"""
import numpy as np
# import sobol_seq
import pygmo as pg
from pygmo import hypervolume
import itertools
#import matplotlib
#import matplotlib.pyplot as plt
# matplotlib.use("Agg")
from scipy.spatial.distance import cdist

from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import mfmes as MFBO
import mfmodel as MFGP
import os
from test_functions import mfbranin,mfCurrin
functions=[mfbranin,mfCurrin]
experiment_name='branin_Currin_2'
d=2
cost = [[1, 10],[1, 10]]



paths=''

#approx='TG'
approx='NI'

fir_num = 5

seed=0
np.random.seed(seed)
sample_number=1
referencePoint = [1e5]*len(functions)
bound=[0,1]
x =np.random.uniform(bound[0], bound[1],size=(1000, d))
# Create data from functions
y=[[] for i in range(len(functions))]
for i in range(len(functions)):
    for m in range(len(cost[i])):
        for xi in x:
            y[i].append(functions[i](xi,d,m))
y=[np.asarray(y[i]) for i in range(len(y))]

# Initial setting
size = np.size(x, 0)
M = [len(i) for i in cost]
fidelity_iter=[np.array([fir_num for j in range(M[i]-1)]+[1]) for i in range(len(M)) ]
#fidelity_iter = [np.array([fir_num, fir_num, 1]),np.array([fir_num, 0])]
total_cost = sum([sum([(float(cost[i][m])/cost[i][M[i]-1])*fidelity_iter[i][m]  for m in range(M[i])]) for i in range(len(M))])
allcosts=[total_cost]
candidate_x = [np.c_[np.zeros(size), x] for i in range(len(functions))]
for i in range(len(functions)):
    for m in range(1, M[i]):
        candidate_x[i] = np.r_[candidate_x[i], np.c_[m*np.ones(size), x]]

# Kernel configuration
kernel_f = 1. * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e2))
kernel_f.set_params(k1__constant_value_bounds=(1., 1.))

kernel_e = 0.1 * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e2))
kernel_e.set_params(k1__constant_value_bounds=(0.1, 0.1))

kernel = MFGP.MFGPKernel(kernel_f, kernel_e)

###################GP Initialisation##########################
GPs=[]
GP_mean=[]
GP_std=[]
cov=[]
MFMES=[]
y_max=[]
GP_index=[]
func_samples=[]
acq_funcs=[]
for i in range(len(functions)):
    GPs.append(MFGP.MFGPRegressor(kernel=kernel))
    GP_mean.append([])
    GP_std.append([])
    cov.append([])
    MFMES.append(0)
    y_max.append(0)
    temp0=[]
    for m in range(M[i]):
        temp0=temp0+list(np.random.randint(size*m, size*(m+1), fidelity_iter[i][m]))
    GP_index.append(np.array(temp0))
#    GP_index.append(np.random.randint(0, size, fir_num))
    func_samples.append([])
    acq_funcs.append([])
experiment_num=0
cost_input_output= open(str(experiment_num)+approx+'_input_output.txt', "a")
print("total_cost:",total_cost)

    
for j in range(300):
    if j%5!=0:
        for i in range(len(functions)):
            GPs[i].fit(candidate_x[i][GP_index[i].tolist()], y[i][GP_index[i].tolist()])
            GP_mean[i], GP_std[i], cov[i] = GPs[i].predict(candidate_x[i])
#            print("Inference Highest fidelity",GP_mean[i][x_best_index+size*(M[i]-1)])

    else:
        for i in range(len(functions)):
            GPs[i].optimized_fit(candidate_x[i][GP_index[i].tolist()], y[i][GP_index[i].tolist()])
            GP_mean[i], GP_std[i], cov[i] = GPs[i].optimized_predict(candidate_x[i])
    #################################################################
    for i in range(len(functions)):
        if fidelity_iter[i][M[i]-1] > 0:
            y_max[i] = np.max(y[i][GP_index[i][GP_index[i]>=(M[i]-1)*size]])
        else:
            y_max[i] = GP_mean[i][(M[i]-1)*size:][np.argmax(GP_mean[i][(M[i]-1)*size:]+GP_std[i][(M[i]-1)*size:])]

        # Acquisition function calculation
    for i in range(len(functions)):
        if approx=='NI':
            MFMES[i] = MFBO.MultiFidelityMaxvalueEntroySearch_NI(GP_mean[i], GP_std[i], y_max[i], GP_index[i], M[i], cost[i], size, cov[i],RegressionModel=GPs[i],sampling_num=sample_number)
        else:
            MFMES[i] = MFBO.MultiFidelityMaxvalueEntroySearch_TG(GP_mean[i], GP_std[i], y_max[i], GP_index[i], M[i], cost[i],size,RegressionModel=GPs[i],sampling_num=sample_number)
        func_samples[i]=MFMES[i].Sampling_RFM()
    max_samples=[]
    for i in range(sample_number):
        front=[[-1*func_samples[k][l][i] for k in range(len(functions))] for l in range(size)] 
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = front)
        cheap_pareto_front=[front[K] for K in ndf[0]]
        maxoffunctions=[-1*min(f) for f in list(zip(*cheap_pareto_front))]
        max_samples.append(maxoffunctions)
    max_samples=list(zip(*max_samples))
        
    for i in range(len(functions)):
        acq_funcs[i]=MFMES[i].calc_acq(np.asarray(max_samples[i]))
    #result[0]values of acq and remaining are the fidelity of each function 
    result=np.zeros((size,len(functions)+1))
    for k in range(size):
        temp=[]
        for i in range(len(functions)):
            temp.append([acq_funcs[i][k+m*size] for m in range(M[i])])
        indecies=list(itertools.product(*[range(len(x)) for x in temp]))
        values_costs=[sum([float(cost[i][m])/cost[i][M[i]-1] for i,m in zip(range(len(functions)),index)]) for index in indecies]
        values=[float(sum(AF))/i for AF,i in zip(list(itertools.product(*temp)),values_costs)]
        result[k][0]=max(values)
        max_index=np.argmax(values)        
        for i in range(len(functions)):
            result[k][i+1]=indecies[max_index][i]
    x_best_index=np.argmax(list(zip(*result))[0])
    for i in range(len(functions)):    
        new_index=int(x_best_index+size*result[x_best_index][i+1])
        print("new_input",candidate_x[i][new_index])                
        GP_index[i] = np.r_[GP_index[i], [new_index]]
        total_cost += float(cost[i][new_index//size])/cost[i][M[i]-1]
        fidelity_iter[i][new_index//size] += 1
        
    cost_input_output.write(str(total_cost)+' '+str(candidate_x[i][new_index])+' '+str(np.array([y[i][new_index] for i in range(len(functions))]))+"\n")
    cost_input_output.close()
    print("total_cost: ",total_cost)
    cost_input_output= open(str(experiment_num)+approx+'_input_output.txt', "a")     
    allcosts.append(total_cost)

cost_input_output.close()
    