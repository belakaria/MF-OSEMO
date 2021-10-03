# -*- coding: utf-8 -*-

from sklearn.kernel_approximation import RBFSampler
import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import sys
import time
class MultiFidelityMaxvalueEntroySearch_NI():
    def __init__(self, mean, std, y_max, index, M, cost, size, cov, sampling_num=10, EntropyApproxNum=100, RegressionModel=0, sigma_epsilon = 1e-3):
        self.mean = np.c_[mean]
        self.std = np.c_[std]
        self.y_max = y_max
        self.index = index
        self.M = M
        self.cost = cost
        self.size = size  
        self.cov = np.c_[cov]
        self.sampling_num = sampling_num
        self.EntropyApproxNum = EntropyApproxNum
        self.RegModel = RegressionModel
        self.x = RegressionModel.x_test
        self.sigma_epsilon = sigma_epsilon

    def next_index(self):
        temp = self.acq_func.copy()
        temp[self.index] = -1e5
        return np.argmax(temp)

    def MF_rbf_fit_transform(self, feature_size=100):
        rbf_features = RBFSampler(gamma=1.0/(2*self.RegModel.kernel.kernel_f.get_params()['k2__length_scale']**2), n_components=feature_size, random_state=0)
        X_test_features = rbf_features.fit_transform(self.x[:self.size,1:])

        coeff = np.tril(np.ones((self.M, self.M)))
        coeff[:,1:] = np.sqrt(0.1)*coeff[:,1:]
        X_test_features = np.kron(coeff, X_test_features)
        X_train_features = X_test_features[self.index, :]

        return X_train_features, X_test_features

    def Sampling_RFM(self):
        rbf_feature_size = 100
        X_train_features, X_test_features = self.MF_rbf_fit_transform(feature_size = rbf_feature_size)
        
        A_inv = np.linalg.inv((X_train_features.T).dot(X_train_features) + np.eye(rbf_feature_size*self.M) / self.RegModel.beta)
        weights_mean = np.ravel(A_inv.dot(X_train_features.T).dot(self.RegModel.y))
        weights_var = A_inv / self.RegModel.beta 

        L = np.linalg.cholesky(weights_var)
        standard_normal_rvs = np.random.normal(0, 1, (np.size(weights_mean), self.sampling_num))
        weights_sample = np.matlib.repmat(np.c_[weights_mean], 1, self.sampling_num) + L.dot(standard_normal_rvs)

        func_sample = X_test_features.dot(weights_sample) * self.RegModel.y_std + self.RegModel.y_mean
        return func_sample[(self.M-1)*self.size :self.M*self.size, :]


    def low_TNEntropy(self, max, high_mean, high_std, low_mean, low_std, cov): 
        inputs = np.linspace(0, 1, self.EntropyApproxNum, endpoint=False)
        DQ_left = norm.ppf(1e-12, loc=low_mean, scale=low_std)
        DQ_right = norm.ppf(1.-1e-12, loc=low_mean, scale=low_std)
        inputs = np.reshape(inputs, (1, 1, self.EntropyApproxNum)) * (DQ_right - DQ_left) + DQ_left
        truncated_constant = norm.cdf((max-high_mean)/high_std)
        truncated_constant[truncated_constant<=0] = 1e-15
        conditionedCDF = norm.cdf((max - (high_mean + cov/np.power(low_std, 2)*(inputs-low_mean))) / np.sqrt(np.power(high_std, 2)-np.power(cov, 2)/np.power(low_std, 2)))
        conditionedCDF[conditionedCDF<=0] = 1e-15
        pdf = norm.pdf(inputs, loc=low_mean, scale=low_std)
        TNEntropy = - conditionedCDF/truncated_constant * pdf * (np.log(conditionedCDF) + np.log(pdf) - np.log(truncated_constant)) * (DQ_right - DQ_left)
        if np.sum(np.isnan(inputs)) > 0:
            print('inputs is error')
        
        if np.sum(np.isnan(DQ_left)) > 0:
            print('DQleft is error')
        
        if np.sum(np.isnan(DQ_right)) > 0:
            print('DQright is error')
        
        if np.sum(np.isnan(truncated_constant)) > 0 or np.sum(truncated_constant<=0) > 0:
            print('truncated_constant is error')
        
        if np.sum(np.isnan(conditionedCDF)) > 0 or np.sum(conditionedCDF<=0) > 0:
            print('conditionedCDF is error')
        
        if np.sum(np.isnan(pdf)) > 0:
            print('pdf is error')
        
        if np.sum(np.isnan(TNEntropy)) > 0:
            print('TNEntropy is error')
        TNEntropy = np.mean(TNEntropy, 2)
        TNEntropy = np.mean(TNEntropy, 1)
        return TNEntropy

    def calc_acq(self,max_sample):
        max_sample[max_sample < self.y_max + 5*np.sqrt(self.sigma_epsilon)] = self.y_max + 5*np.sqrt(self.sigma_epsilon)
        self.max_samples = max_sample
        normalized_max = (max_sample - np.c_[self.mean[(self.M-1)*self.size:]]) /np.c_[self.std[(self.M-1)*self.size:]]

        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)

        high_acq_func = (normalized_max * pdf) / (2*cdf) - np.log(cdf)
        high_acq_func = np.mean(high_acq_func, 1)
        max_sample = np.c_[max_sample].T

        high_mean = np.matlib.repmat(self.mean[(self.M-1)*self.size:], self.M-1, 1)
        high_std = np.matlib.repmat(self.std[(self.M-1)*self.size:], self.M-1, 1)

        low_TN_entropy = self.low_TNEntropy(np.atleast_3d(max_sample), np.atleast_3d(high_mean), np.atleast_3d(high_std), np.atleast_3d(
            self.mean[:(self.M-1)*self.size]), np.atleast_3d(self.std[: (self.M-1)*self.size]), np.atleast_3d(self.cov))
        low_entropy = np.ravel(np.log(np.sqrt(2*np.pi*np.e)*self.std[: (self.M-1)*self.size]))
        low_acq_func = low_entropy - low_TN_entropy

        Digit_max = math.ceil(np.log10(np.max(low_acq_func[:self.size])))
        Digit_min = math.ceil(np.log10(np.min(low_acq_func[:self.size])))
        if Digit_max < Digit_min+2:
            print('ERROR: digit of Devide Quadrature error is danger')
            print('MaxDigit'+str(Digit_max)+', MinDigit'+str(Digit_min))

        acq_func = np.r_[low_acq_func, high_acq_func]

        for m in range(0, self.M-1):
            acq_func[self.size*m: self.size*(m+1)] = acq_func[self.size*m: self.size*(m+1)]
        acq_func[self.index] = -1e100
        return acq_func



class MultiFidelityMaxvalueEntroySearch_TG():
    def __init__(self, mean, std, y_max, index, M, cost, size, RegressionModel, sampling_num=10):
        self.mean = np.c_[mean]
        self.std = np.c_[std]
        self.y_max = y_max
        self.index = index
        self.M = M
        self.cost = cost
        self.size = size  
        self.RegModel = RegressionModel
        self.x = RegressionModel.x_test
        self.sampling_num = sampling_num

    def next_index(self):
        temp = self.acq_func.copy()
        temp[self.index] = -1e5
        return np.argmax(temp)

    def MF_rbf_fit_transform(self, feature_size=100):
        rbf_features = RBFSampler(gamma=1.0/(2*self.RegModel.kernel.kernel_f.get_params()['k2__length_scale']**2), n_components=feature_size, random_state=0)
        X_test_features = rbf_features.fit_transform(self.x[:self.size,1:])

        coeff = np.tril(np.ones((self.M, self.M)))
        coeff[:,1:] = np.sqrt(0.1)*coeff[:,1:]
        X_test_features = np.kron(coeff, X_test_features)

        X_train_features = X_test_features[self.index, :]

        return X_train_features, X_test_features

    def Sampling_RFM(self):
        rbf_feature_size = 100
        X_train_features, X_test_features = self.MF_rbf_fit_transform(feature_size = rbf_feature_size)
        
        A_inv = np.linalg.inv((X_train_features.T).dot(X_train_features) + np.eye(rbf_feature_size*self.M) / self.RegModel.beta)
        weights_mean = np.ravel(A_inv.dot(X_train_features.T).dot(self.RegModel.y))
        weights_var = A_inv / self.RegModel.beta 

        L = np.linalg.cholesky(weights_var)
        standard_normal_rvs = np.random.normal(0, 1, (np.size(weights_mean), self.sampling_num))
        weights_sample = np.matlib.repmat(np.c_[weights_mean], 1, self.sampling_num) + L.dot(standard_normal_rvs)

        func_sample = X_test_features.dot(weights_sample) * self.RegModel.y_std + self.RegModel.y_mean
        return func_sample[(self.M-1)*self.size :self.M*self.size, :]

    def calc_acq(self,max_sample):

        c = np.zeros(self.M)
        max_sample[max_sample < self.y_max + 5*np.sqrt(1.0/self.RegModel.beta)] = self.y_max + 5*np.sqrt(1.0/self.RegModel.beta)
        max_sample = (np.c_[max_sample] + np.c_[c].T).T
        temp = np.matlib.repmat(np.c_[max_sample[0]].T, self.size, 1)
        for m in range(1, self.M):
            temp = np.r_[temp, np.matlib.repmat(np.c_[max_sample[m]].T, self.size, 1)]
        max_sample = temp
        normalized_max = (max_sample - np.c_[self.mean]) / np.c_[self.std]

        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)

        acq_func = (normalized_max * pdf) / (2*cdf) - np.log(cdf)
        acq_func = np.mean(acq_func, 1)
        for m in range(0, self.M-1):
            acq_func[self.size*m: self.size *(m+1)] = acq_func[self.size*m: self.size*(m+1)]
        acq_func[self.index] = -1e100
        return acq_func

