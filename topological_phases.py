#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:07:16 2024

@author: gabriel
"""

from superconductor import SpinOrbitSuperconductorKY
import numpy as np
import matplotlib.pyplot as plt

k_y = 0   
L_x = 50
t = 10
mu_values = np.linspace(-60, 60, 100)
Delta_s = 0.2
Lambda = 0.56
theta = np.pi/2
B = 0.4
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
B_z = 0
E_mu = np.zeros((len(mu_values), 4*L_x))

for i, mu in enumerate(mu_values):
    for j in range(4*L_x):
        H = SpinOrbitSuperconductorKY(k_y, L_x, t, mu, Delta_s, Lambda, B_x, B_y, B_z)
        E_mu[i, j] = np.linalg.eigvalsh(H.matrix)[j]
    print(i)
fig, ax = plt.subplots()
for i in range(4*L_x):
    ax.plot(mu_values, E_mu[:, i])

ax.set_xlabel(r"$\mu$")
ax.set_ylabel(r"$E(k_y=$"+f"{np.round(k_y, 2)}")
