#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 17:38:59 2024

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt

def G(k_n, E, mu):
    return 1/(1j*k_n-(E-mu))

def Matsubara_sum(k_n, E, mu, beta):
    sumand = []
    for k_n_value in k_n:
        sumand.append(G(k_n_value, E, mu)*np.exp(1j*k_n_value*0.001))
    sumand = 1/beta*np.array(sumand)
    return np.sum(sumand)

beta = 10
N = 3000
mu = 0
E = np.linspace(-5, 2, 300)
k_n = np.pi/(beta)*(2*np.arange(-N, N)+1)
# k_n = np.delete(k_n, N)

M = [Matsubara_sum(k_n, E_value, mu, beta) for E_value in E]
fig, ax = plt.subplots()
ax.plot(E, M)

