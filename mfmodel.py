# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:57:25 2019

@author: Syrine Belakaria
This code is based on the code from https://github.com/takeno1995/BayesianOptimization
"""
import numpy as np


class MFGPRegressor(object):
    def __init__(self, kernel, beta=1e3):
        self.kernel = kernel
        self.beta = beta

    def normalize(self, y):
        self.y_mean = np.mean(y)
        self.y_std = np.sqrt(np.mean((y - self.y_mean) ** 2))
        return np.c_[(y - self.y_mean) / self.y_std]

    def fit(self, x, y):
        self.x_train = np.c_[x]
        self.y = self.normalize(y)
        Gram = self.kernel(self.x_train, self.x_train)
        covariance = Gram + np.identity(np.size(self.x_train, 0)) / self.beta
        self.precision = np.linalg.inv(covariance)

    def optimized_fit(self, x, y, MT=False, error_opt=True):
        self.x_train = np.c_[x]
        self.y = self.normalize(y)

        if MT:
            error_min = self.kernel.kernel_e.get_params()["k1__constant_value_bounds"][
                0
            ]
            error_max = self.kernel.kernel_e.get_params()["k1__constant_value_bounds"][
                1
            ]
            ell_min = self.kernel.kernel_f.get_params()["k2__length_scale_bounds"][0]
            ell_max = self.kernel.kernel_f.get_params()["k2__length_scale_bounds"][1]
            initial_error = self.kernel.kernel_e.get_params()["k1__constant_value"]
        else:
            error_min = self.kernel.kernel_z.get_params()["length_scale_bounds"][0]
            error_max = self.kernel.kernel_z.get_params()["length_scale_bounds"][1]
            ell_min = self.kernel.kernel_f.get_params()["k2__length_scale_bounds"][0]
            ell_max = self.kernel.kernel_f.get_params()["k2__length_scale_bounds"][1]
            initial_error = self.kernel.kernel_z.get_params()["length_scale"]

        selected_sigma_f = 1
        self.evidence = -np.inf

        ell_range = np.exp(np.linspace(np.log(ell_min), np.log(ell_max), 100))
        error_range = np.exp(np.linspace(np.log(error_min), np.log(error_max), 10))

        if error_opt:
            for ell in ell_range:
                for error in error_range:
                    self.kernel.set_params(selected_sigma_f, error, ell)

                    Gram = self.kernel(self.x_train, self.x_train)
                    covariance = Gram + np.identity(np.size(x, 0)) / self.beta
                    L = np.linalg.cholesky(covariance)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
                    evidence = -(self.y.T).dot(alpha) / 2 - np.sum(np.log(np.diag(L)))
                    if evidence > self.evidence:
                        selected_error = error
                        selected_ell = ell
                        self.L = L
                        self.alpha = alpha
                        self.evidence = evidence
            self.kernel.set_params(selected_sigma_f, selected_error, selected_ell)
        else:
            for ell in ell_range:
                self.kernel.set_params(selected_sigma_f, initial_error, ell)

                Gram = self.kernel(self.x_train, self.x_train)
                covariance = Gram + np.identity(np.size(x, 0)) / self.beta
                L = np.linalg.cholesky(covariance)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
                evidence = -(self.y.T).dot(alpha) / 2 - np.sum(np.log(np.diag(L)))
                if evidence > self.evidence:
                    selected_ell = ell
                    self.L = L
                    self.alpha = alpha
                    self.evidence = evidence

            self.kernel.set_params(selected_sigma_f, initial_error, selected_ell)

    def predict(self, x, cov=True, var=False):
        self.M = np.int(np.max(x[:, 0]) + 1)
        self.size = np.int(np.size(x[:, 0]) / self.M)
        self.x_test = np.c_[x]

        K = self.kernel(self.x_test, self.x_train)
        temp = K.dot(self.precision)
        self.mean = temp.dot(self.y).ravel()  # 平均

        if var:
            var = self.kernel(self.x_test, self.x_test) - temp.dot(K.T)
            return self.mean * self.y_std + self.y_mean, var * self.y_std**2

        var = self.kernel.diag(self.x_test) - np.einsum("ij,ji->i", temp, K.T)
        if np.min(var) < -1.0 / self.beta or np.sum(np.isnan(var)) > 0:
            print("Error: Var have NAN or minus value")
        elif np.min(var) <= 0:
            print("Var have smaller minus value and substitute small plus value")
            var[var <= 0] = 1e-5
        self.std = np.sqrt(var).ravel()

        if np.min(self.std) <= 0:
            print("ERROR: std is lower than 0")
        if np.sum(np.isnan(self.std)) > 0:
            print("Error: std have NAN")

        if cov:
            cov_k = np.array([])
            for m in range(0, self.M - 1):
                cov = self.kernel.diag(
                    self.x_test[m * self.size : (m + 1) * self.size]
                ) - np.einsum(
                    "ij,ji->i",
                    temp[(self.M - 1) * self.size :, :],
                    K.T[:, m * self.size : (m + 1) * self.size],
                )
                cov_k = np.r_[cov_k, cov]
            return (
                self.mean * self.y_std + self.y_mean,
                self.std * self.y_std,
                cov_k * np.power(self.y_std, 2),
            )
        else:
            return self.mean * self.y_std + self.y_mean, self.std * self.y_std

    def optimized_predict(self, x, cov=True):
        self.M = np.int(x[-1, 0] + 1)
        self.size = np.size(x[:, 0]) // self.M
        self.x_test = np.c_[x]

        K = self.kernel(self.x_train, self.x_test)
        v = np.linalg.solve(self.L, K)
        self.mean = K.T.dot(self.alpha).ravel()

        var = self.kernel.diag(self.x_test) - np.einsum("ij,ji->i", v.T, v)
        if np.min(var) < -1.0 / self.beta or np.sum(np.isnan(var)) > 0:
            print("Error: Var have NAN or minus value")
        elif np.min(var) <= 0:
            print("Var have smaller minus value and substitute small plus value")
            var[var <= 0] = 1e-5
        self.std = np.sqrt(var).ravel()

        if cov:
            cov_k = np.array([])
            for m in range(0, self.M - 1):
                cov = self.kernel.diag(
                    self.x_test[m * self.size : (m + 1) * self.size, :]
                ) - np.einsum(
                    "ij,ji->i",
                    v.T[(self.M - 1) * self.size :, :],
                    v[:, m * self.size : (m + 1) * self.size],
                )
                cov_k = np.r_[cov_k, cov]
            return (
                self.mean * self.y_std + self.y_mean,
                self.std * self.y_std,
                cov_k * np.power(self.y_std, 2),
            )
        else:
            return self.mean * self.y_std + self.y_mean, self.std * self.y_std


# Multi-fidelity Kernel
class MFGPKernel(object):
    def __init__(self, kernel_f, kernel_e):
        self.kernel_f = kernel_f
        self.kernel_e = kernel_e

    def set_params(self, sigma_f, error_sigma, ell):
        self.kernel_f.set_params(k1__constant_value=sigma_f, k2__length_scale=ell)
        self.kernel_e.set_params(k1__constant_value=error_sigma, k2__length_scale=ell)

    def calc_diff(self, x1, x2):
        error_constant = np.concatenate(
            [
                np.c_[x1[:, 0]].dot(np.c_[np.ones(np.size(x2, 0))].T),
                np.c_[np.ones(np.size(x1, 0))].dot(np.c_[(x2[:, 0])].T),
            ],
            axis=0,
        ).reshape(2, np.size(x1, 0), np.size(x2, 0))
        error_constant = np.amin(error_constant, 0)
        K_f = self.kernel_f(x1[:, 1:], x2[:, 1:])
        K_e = self.kernel_e(x1[:, 1:], x2[:, 1:])
        return K_f + error_constant * K_e

    def diag(self, x):
        error_constant = x[:, 0]
        K_f = self.kernel_f.diag(x[:, 1:])
        K_e = self.kernel_e.diag(x[:, 1:])
        return K_f + error_constant * K_e

    def __call__(self, x1, x2):
        K = self.calc_diff(x1, x2)
        return K

    def set_length_scale(self, X):
        x = np.atleast_2d(X)
        m, _ = x.shape
        norm = np.dot(np.ones([m, 1]), np.atleast_2d(np.sum(x**2, axis=1)))
        norms = norm + norm.T - -2 * x.dot(x.T)
        norms = np.sort(norms, axis=1)
        Standard_length_scale = np.sqrt(np.median(np.mean(norms[:, 1:11])))
        print("Standard_length_scale:", Standard_length_scale)
        self.kernel_f.set_params(
            k2__length_scale=Standard_length_scale,
            k2__length_scale_bounds=(
                1e-2 * Standard_length_scale,
                1e1 * Standard_length_scale,
            ),
        )
        self.kernel_e.set_params(
            k2__length_scale=Standard_length_scale,
            k2__length_scale_bounds=(
                1e-2 * Standard_length_scale,
                1e1 * Standard_length_scale,
            ),
        )
