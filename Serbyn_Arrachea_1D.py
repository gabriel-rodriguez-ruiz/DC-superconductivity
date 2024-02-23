#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:48:33 2024

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from Serbyn_1D import get_Q, get_Q_k, integrand_Serbyn
from Superconductor_1D_without_SOC import get_sigma, Rho_k, Fermi_function_derivative, get_sigma_quad, get_sigma_k,get_sigma_quad_k, integrand, get_zero_order_k

beta = 100
N = 1000
L = 1000
k = 2*np.pi/L*np.arange(0, L)
w_0 = 1
Gamma = 0.1
Delta = 0
mu = 0
B_x = 0
B_y = 0
omega = np.linspace(-2, 2, 1000)
# epsilon_n = 2*np.pi/beta*(np.arange(-N, N)+1/2)
epsilon_n = np.pi/beta*(np.arange(-N, N))

# Q = get_Q(k, w_0, Gamma, Delta, mu, N, beta)
# sigma_quad = get_sigma_quad(k, w_0, Gamma, B_x, B_y, Delta, mu, beta)
# sigma_trapz = get_sigma(k, omega, w_0, Gamma, B_x, B_y, Delta, mu, beta)

k = 1
Q_k = get_Q_k(k, w_0, Gamma, Delta, mu, N, beta)
sigma_quad_k = get_sigma_quad_k(k, w_0, Gamma, B_x, B_y, Delta, mu, beta)
sigma_k = get_sigma_k(k, omega, w_0, Gamma, B_x, B_y, Delta, mu, beta)


fig, ax = plt.subplots()
ax.plot(omega, [np.trace(integrand(k, omega, w_0, Gamma, B_x, B_y, Delta, mu, beta)) for omega in omega], "o")
ax.plot(epsilon_n, [integrand_Serbyn(k, epsilon_n, w_0, Gamma, Delta, mu) for epsilon_n in epsilon_n], "o")

zero_order_k = get_zero_order_k(k, omega, w_0, Gamma, B_x, B_y, Delta, mu, beta)