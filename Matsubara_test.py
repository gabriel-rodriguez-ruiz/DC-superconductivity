#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 17:38:59 2024

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt

def G(k_n, E, mu, Gamma):
    return 1/(1j*k_n-(E-mu)-Gamma)

def Matsubara_sum(k_n, E, mu, beta, Gamma):
    sumand = np.zeros_like(k_n, dtype=np.clongdouble)
    for i in range(len(k_n)):
        sumand[i] = G(k_n[i], E, mu, Gamma)*np.exp(1j*k_n[i]*0.01)
    return 1/beta*np.sum(sumand, dtype=np.clongdouble)

beta = np.linspace(10, 100, 11)
N = 100000
mu = 0
Gamma = 0
E = np.linspace(-5, 2, 50)
# k_n = np.pi/beta*(2*np.arange(-N, N, dtype=np.int64)+1)

# M = [Matsubara_sum(k_n, E_value, mu, beta, Gamma) for E_value in E]
# fig, ax = plt.subplots()
# ax.plot(E, M)

E_value = -1
N = np.linspace(1, 100000, 20)
fig, ax = plt.subplots()

for beta_value in beta:
    M = [Matsubara_sum(np.pi/beta_value*(2*np.arange(-N_value, N_value, dtype=np.int64)+1), E_value, mu, beta_value, Gamma) for N_value in N]
    ax.plot(N, M, label=r"$\beta=$"+f"{beta}")