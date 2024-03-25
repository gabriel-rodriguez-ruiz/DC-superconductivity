#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:39:20 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x
import scipy.integrate
import matplotlib.pyplot as plt

def G_k(k_x, k_y, omega, w_0, Gamma, B_x, B_y, Delta, mu, Lambda):
    r""" Frozen Green function.
    
    .. math ::
        w_k(t=0)= (2 w_0 \cos(k) - \mu ) \tau_z\sigma_0
        
        \lambda_{k_x} = 2\lambda \sin(k_x)\tau_0\sigma_y
        
        \lambda_{k_y} = -2\lambda \sin(k_y)\tau_0\sigma_x
        
        G^f_{k}(t=0,\omega)=\left[\omega\tau_0\sigma_0-(w_{k}(t=0) -\lambda_{k_x}-\lambda_{k_y} -B_x\tau_0\sigma_x-B_y\tau_0\sigma_y +\Delta\tau_x\sigma_0)+i \Gamma\tau_0\sigma_0 \right]^{-1}
    
    """
    Lambda_x = 2*Lambda*np.sin(k_x)*np.kron(tau_0, sigma_y)
    Lambda_y = -2*Lambda*np.sin(k_y)*np.kron(tau_0, sigma_x)
    H_k = (
        (2*w_0*np.cos(k_x)+2*w_0*np.cos(k_y)-mu)*np.kron(tau_z, sigma_0)
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
        + Delta*np.kron(tau_x, sigma_0)
        + Lambda_x + Lambda_y
           )
    return np.linalg.inv(omega*np.kron(tau_0, sigma_0) - H_k + 1j*np.sign(omega)*Gamma*np.kron(tau_0, sigma_0))

def get_Q_k(k_x, k_y, w_0, Gamma, B_x, B_y, Delta, mu, Lambda, N, beta):
    r""" Kernel for given k=(k_x, k_y)
        Returns a 2D-array.
    
    .. math ::
        Q_{\alpha, \beta}(k_x, k_y) = \frac{-1}{2\beta}\sum_{i\epsilon_n}\left(Tr\left[v_\alpha G(\mathbf{k},i\epsilon_n)v_\beta G(\mathbf{k},i\epsilon_n)\right]\right) 
    """
    epsilon_n = np.pi/beta * (2*np.arange(-N, N) + 1)
    sumand = np.zeros((len(epsilon_n), 2, 2), dtype=complex)
    v_x = (
           -2*w_0*np.sin(k_x) * np.kron(tau_z, sigma_0)
           +2*Lambda*np.cos(k_x) * np.kron(tau_z, sigma_y)
           )
    v_y = (
           -2*w_0*np.sin(k_y) * np.kron(tau_z, sigma_0)
           -2*Lambda*np.cos(k_y) * np.kron(tau_z, sigma_x)
           )
    for i in range(len(epsilon_n)):
        G = G_k(
                k_x, k_y, 1j*epsilon_n[i], w_0, Gamma,
                B_x, B_y, Delta, mu, Lambda
                )
        sumand[i, :, :] = np.array([[np.trace(v_x @ G @ v_x @ G), np.trace(v_x @ G @ v_y @ G)],
                              [np.trace(v_y @ G @ v_x @ G), np.trace(v_y @ G @ v_y @ G)]])
    return -1/(2*beta) * np.sum(sumand, dtype=complex, axis=0)

def get_Q(k_x, k_y, w_0, Gamma, B_x, B_y, Delta, mu, Lambda, N, beta, Alpha, Beta):
    L_x = len(k_x)
    L_y = len(k_y)
    sumand = np.zeros((2, 2, L_x, L_y), dtype=complex)
    for i in Alpha:
        for j in Beta:
            for k in range(len(k_x)):
                for l in range(len(k_y)):
                    sumand[i, j, k, l] = get_Q_k(k_x[k], k_y[l], w_0, Gamma,
                                           B_x, B_y, Delta, mu, Lambda, N, beta)[i, j]
    return 1/(L_x*L_y) * np.sum(sumand, dtype=complex, axis=(2,3))

#%%
beta = 10
N = 100
Gamma = 0.01
mu = 0
w_0 = 1
Delta = 0
B_x = 0
B_y = 0
Lambda = 0

L = 30
k_x = 2*np.pi/L*np.arange(0, L/2)   #1/4 of the Brillouin zone
k_y = 2*np.pi/L*np.arange(0, L/2)
K_x, K_y = np.meshgrid(k_x, k_y)

Alpha = [0]
Beta = [0]

Z = np.zeros((len(Alpha), len(Beta), len(k_x), len(k_y)), dtype=complex)
for i in range(len(Alpha)):
    for j in range(len(Beta)):
        for k, k_x_value in enumerate(k_x):
            print(k)
            for l, k_y_value in enumerate(k_y):
                Z[i, j, k, l] = 1/L**2 * get_Q_k(k_x_value, k_y_value, w_0, Gamma,
                                           B_x, B_y, Delta, mu, Lambda, N, beta)[i, j]

fig, ax = plt.subplots()
X, Y = np.meshgrid(k_x, k_y)
cs = ax.contourf(X, Y, Z[0, 0, :, :])
cbar = fig.colorbar(cs)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")

# Q = get_Q(k_x, k_y, w_0, Gamma, B_x, B_y, Delta, mu, Lambda, N, beta)