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
        
        \lambda_{k_x} = 2\lambda \sin(k_x)\tau_z\sigma_y
        
        \lambda_{k_y} = -2\lambda \sin(k_y)\tau_z\sigma_x
        
        G^f_{k}(t=0,\omega)=\left[\omega\tau_0\sigma_0-(w_{k}(t=0) -\lambda_{k_x}-\lambda_{k_y} -B_x\tau_0\sigma_x-B_y\tau_0\sigma_y +\Delta\tau_x\sigma_0)+i \Gamma\tau_0\sigma_0 \right]^{-1}
    
    """
    Lambda_x = 2*Lambda*np.sin(k_x)*np.kron(tau_z, sigma_y)
    Lambda_y = -2*Lambda*np.sin(k_y)*np.kron(tau_z, sigma_x)
    H_k = (
        (-2*w_0*np.cos(k_x)-2*w_0*np.cos(k_y)-mu)*np.kron(tau_z, sigma_0)
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
        + Delta*np.kron(tau_x, sigma_0)
        + Lambda_x + Lambda_y
           )
    return np.linalg.inv(omega*np.kron(tau_0, sigma_0) - H_k + 1j*np.sign(omega)*Gamma*np.kron(tau_0, sigma_0))

def get_Q_k(k_x, k_y, w_0, Gamma, B_x, B_y, Delta, mu, Lambda, N, beta, Alpha, Beta):
    r""" Kernel for given k=(k_x, k_y)
        Returns a 2D-array.
    
    .. math ::
        Q_{\alpha, \beta}(k_x, k_y) = \frac{-1}{2\beta}\sum_{i\epsilon_n}\left(Tr\left[v_\alpha G(\mathbf{k},i\epsilon_n)v_\beta G(\mathbf{k},i\epsilon_n)\right]\right) 
    """
    epsilon_n = np.pi/beta * (2*np.arange(-N, N) + 1)
    sumand = np.zeros((len(epsilon_n)), dtype=complex)
    v_x = (
           -2*w_0*np.sin(k_x) * np.kron(tau_z, sigma_0)
           +2*Lambda*np.cos(k_x) * np.kron(tau_0, sigma_y)
           )
    v_y = (
           -2*w_0*np.sin(k_y) * np.kron(tau_z, sigma_0)
           -2*Lambda*np.cos(k_y) * np.kron(tau_0, sigma_x)
           )
    for i in range(len(epsilon_n)):
        G = G_k(
                k_x, k_y, 1j*epsilon_n[i], w_0, Gamma,
                B_x, B_y, Delta, mu, Lambda
                )
        if [Alpha, Beta]==[0,0]:
            sumand[i] = np.trace(v_x @ G @ v_x @ G)
        elif [Alpha, Beta]==[0,1]:
            sumand[i] = np.trace(v_x @ G @ v_y @ G)
        elif [Alpha, Beta]==[1,0]:
            sumand[i] = np.trace(v_y @ G @ v_x @ G)
        else:
            sumand[i] = np.trace(v_y @ G @ v_y @ G)
    return -1/(2*beta) * np.sum(sumand, dtype=complex, axis=0)

def get_Q(k_x, k_y, w_0, Gamma, B_x, B_y, Delta, mu, Lambda, N, beta, Alpha, Beta):
    L_x = len(k_x)
    L_y = len(k_y)
    sumand = np.zeros((L_x, L_y), dtype=complex)
    for i in range(len(k_x)):
        for j in range(len(k_y)):
            sumand[i, j] = get_Q_k(k_x[i], k_y[j], w_0, Gamma,
                                   B_x, B_y, Delta, mu, Lambda, N, beta, Alpha, Beta)
    return 1/(L_x*L_y) * np.sum(sumand, dtype=complex)

#%%
beta = 10
N = 100
Gamma = 0.01
mu = 0
w_0 = 1
Delta = 0.1
B_x = 0
B_y = 0
Lambda = 0

L = 30
k_x = 2*np.pi/L*np.arange(0, L/2)   #1/4 of the Brillouin zone
k_y = 2*np.pi/L*np.arange(0, L/2)
K_x, K_y = np.meshgrid(k_x, k_y)

Alpha = 0
Beta = 1

Z = np.zeros((len(k_x), len(k_y)), dtype=complex)

for k, k_x_value in enumerate(k_x):
    print(k)
    for l, k_y_value in enumerate(k_y):
        Z[k, l] = 1/L**2 * (get_Q_k(k_x_value, k_y_value, w_0, Gamma,
                                   B_x, B_y, Delta, mu, Lambda, N, beta, Alpha, Beta)
                            -get_Q_k(k_x_value, k_y_value, w_0, Gamma,
                                                       B_x, B_y, 0, mu, Lambda, N, beta, Alpha, Beta))

fig, ax = plt.subplots()
X, Y = np.meshgrid(k_x, k_y)
# cs = ax.contourf(X, Y, Z)
im = ax.imshow(np.real(Z), extent=[X.min(), X.max(), Y.min(), Y.max()], origin="lower")
fig.colorbar(im)
# cbar = fig.colorbar(cs)
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
# ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
#               ["0", r"$\frac{\pi}{4}$",r"$\frac{\pi}{2}$",
#                r"$\frac{3\pi}{4}$", r"$\pi$"])


#%%
beta = 10
N = 1000
Gamma = 0.01
mu = 0
w_0 = 1
Delta = 0.1
theta = np.pi/2
B = 0.05
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.1
Alpha = 0
Beta = 0

L = 50
k_x = 2*np.pi/L*np.arange(0, L)
k_y = 2*np.pi/L*np.arange(0, L)

Q = get_Q(k_x, k_y, w_0, Gamma, B_x, B_y, Delta, mu, Lambda, N, beta, Alpha, Beta)

#%% Q vs. L

beta = 10
N = 100
Gamma = 0.01
mu = 0
w_0 = 1
Delta = 0.1
theta = np.pi/2
B = 0.05
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.1
Alpha = 0
Beta = 0

L_values = np.linspace(10, 200, 10, dtype=int)

n_L = np.zeros(len(L_values))
for i, L in enumerate(L_values):
    k_x = 2*np.pi/L*np.arange(0, L)
    k_y = 2*np.pi/L*np.arange(0, L)
    # n_L[i] = -(
    #           get_Q(k_x, k_y, w_0, Gamma, B_x, B_y, Delta, mu, Lambda, N, beta, Alpha, Beta)
    #           - get_Q(k_x, k_y, w_0, Gamma, B_x, B_y, 0, mu, Lambda, N, beta, Alpha, Beta)
    #           )
    n_L[i] = -(
              get_Q(k_x, k_y, w_0, Gamma, B_x, B_y, Delta, mu, Lambda, N, beta, Alpha, Beta)
              - get_Q(k_x, k_y, w_0, Gamma, B_x, B_y, 0, mu, Lambda, N, beta, Alpha, Beta)
              )
    print(i)
    
fig, ax = plt.subplots()
ax.plot(L_values, n_L, "o")
ax.set_xlabel("L")
ax.set_ylabel(r"$Q_{xx}$")
ax.set_title(r"$\lambda=$" + f"{Lambda}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             +f"; B={B}")
plt.tight_layout()

#%% Saving
np.savez("Large_L_limit_for Q", L_values=L_values, n_L=n_L, Lambda=Lambda,
         Delta=Delta, B=B, theta=theta)

#%% Q vs. B

beta = 100
N = 10
Gamma = 0.01
w_0 = -10
Delta = 0.2
mu = 2*(20*Delta+2*w_0)
theta = np.pi/2
B_values = np.linspace(0, 3*Delta, 10)
Lambda = 5*Delta/np.sqrt((-4*w_0 + mu)/(-w_0))
Alpha = 0
Beta = 0
L = 10
k_x = 2*np.pi/L*np.arange(0, L)
k_y = 2*np.pi/L*np.arange(0, L)

n_B_y = np.zeros(len(B_values))
for i, B in enumerate(B_values):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    n_B_y[i] = -(
              get_Q(k_x, k_y, w_0, Gamma, B_x, B_y, Delta, mu, Lambda, N, beta, Alpha, Beta)
              - get_Q(k_x, k_y, w_0, Gamma, B_x, B_y, 0, mu, Lambda, N, beta, Alpha, Beta)
              )
    print(i)
    
fig, ax = plt.subplots()
ax.plot(B_values/Delta, n_B_y, "o")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$n(B_y)$")
plt.tight_layout()