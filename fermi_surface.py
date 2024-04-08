#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:04:32 2024

@author: gabriel
"""
import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
import matplotlib.pyplot as plt

def get_Hamiltonian_without_superconductivity(k_x, k_y, w_0, mu, B_x, B_y, Lambda):
    """ Periodic Hamiltonian in x and y with flux.
    """
    return (
            -2*w_0*(np.cos(k_x) + np.cos(k_y)) * sigma_0
            + 2*Lambda * (np.sin(k_x)*sigma_y - np.sin(k_y)*sigma_x)
            - B_x*sigma_x - B_y*sigma_y
            )
def get_Energy_without_superconductivity(k_x_values, k_y_values, w_0, mu, B_x, B_y, Lambda):
    """
    """
    energies = np.zeros((len(k_x_values), len(k_y_values), 2))
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for m in range(2):
                H = get_Hamiltonian_without_superconductivity(k_x, k_y, w_0, mu, B_x, B_y, Lambda)
                energies[i, j, m] = np.linalg.eigvalsh(H)[m]
    return energies

#%%
L_x = 1000
L_y = 1
w_0 = 10
Delta = 0.2
mu = 2*(20*Delta-2*w_0)
theta = np.pi/2
B = 3*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.1#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
k_x_values = np.pi/L_x*np.arange(-L_x, L_x)
k_y_values = [0]

E = get_Energy_without_superconductivity(k_x_values, k_y_values, w_0, mu, B_x, B_y, Lambda)

fig, ax = plt.subplots()
ax.plot(k_x_values, E[:, 0, 0])
ax.plot(k_x_values, E[:, 0, 1])
ax.plot(k_x_values, mu*np.ones_like(k_x_values), "--")

ax.set_ylabel(r"$\epsilon(k_y=0)$")
ax.set_xlabel(r"$k_x$")
ax.set_title(f"w_0={w_0}; mu={mu}; Lambda={Lambda:.2}; k_y=0; Bx={B_x:.2}; By={B_y:.2}")
plt.tight_layout()