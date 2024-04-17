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

def get_Hamiltonian_without_superconductivity(k_x, k_y, w_0, B_x, B_y, Lambda):
    """ Periodic Hamiltonian in x and y with flux.
    """
    return (
            -2*w_0*(np.cos(k_x) + np.cos(k_y)) * sigma_0
            + 2*Lambda * (np.sin(k_x)*sigma_y - np.sin(k_y)*sigma_x)
            - B_x*sigma_x - B_y*sigma_y
            )
def get_Energy_without_superconductivity(k_x_values, k_y_values, w_0, B_x, B_y, Lambda):
    """
    """
    energies = np.zeros((len(k_x_values), len(k_y_values), 2))
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for m in range(2):
                H = get_Hamiltonian_without_superconductivity(k_x, k_y, w_0, B_x, B_y, Lambda)
                energies[i, j, m] = np.linalg.eigvalsh(H)[m]
    return energies

def get_Green_function(omega, k_x_values, k_y_values, w_0, B_x, B_y, Lambda):
    G = np.zeros((len(k_x_values), len(k_y_values), 2, 2), dtype=complex)
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for k in range(2):
                for l in range(2):
                    G[i, j, k, l] = np.linalg.inv(omega*sigma_0 -
                                    get_Hamiltonian_without_superconductivity(k_x, k_y, w_0, B_x, B_y, Lambda))[k, l]
    return G

def get_DOS(omega_values, eta, L_x, L_y, w_0, B_x, B_y, Lambda):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    E = get_Energy_without_superconductivity(k_x_values, k_y_values, w_0, B_x, B_y, Lambda)
    DOS = np.zeros(len(omega_values))
    for i, omega in enumerate(omega_values):
        DOS[i] = 1/(L_x*L_y) * (-1)/np.pi*np.sum(np.imag(1/(omega-E+1j*eta)))
    return DOS
    
def get_normal_density(omega_values, mu, eta, L_x, L_y, w_0, B_x, B_y, Lambda):
    DOS = get_DOS(omega_values, eta, L_x, L_y, w_0, B_x, B_y, Lambda)
    return np.sum(DOS[omega_values<mu])*np.diff(omega_values)[0]

#%% Plot energy bands
L_x = 100
L_y = 100
w_0 = 10
mu = -38 #2*(20*Delta-2*w_0)
theta = np.pi/2
B = 0.6
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.56#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
k_x_values = np.pi/L_x*np.arange(-L_x, L_x)
k_y_values = [0]

E = get_Energy_without_superconductivity(k_x_values, k_y_values, w_0, B_x, B_y, Lambda)

fig, ax = plt.subplots()
ax.plot(k_x_values, E[:, 0, 0])
ax.plot(k_x_values, E[:, 0, 1])
ax.plot(k_x_values, mu*np.ones_like(k_x_values), "--")

ax.set_ylabel(r"$\epsilon(k_y=0)$")
ax.set_xlabel(r"$k_x$")
ax.set_title(f"w_0={w_0}; mu={mu}; Lambda={Lambda:.2}; k_y=0; theta={theta:.2}; B={B:.2}")
plt.tight_layout()

#%% Density of states
L_x = 200
L_y = 200
w_0 = 10
mu = -32
theta = np.pi/2
B = 0
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.56
omega_values = np.linspace(-6*w_0, 6*w_0, 10000)
eta = 0.1


DOS = get_DOS(omega_values, eta, L_x, L_y, w_0, B_x, B_y, Lambda)
S = np.sum(DOS)*np.diff(omega_values)[0]
M = np.sum(DOS[omega_values<mu])*np.diff(omega_values)[0]

fig, ax = plt.subplots()
ax.plot(omega_values, DOS)
# ax.plot(omega_values, DOS[:, 1, 1])

ax.set_ylabel(r"$\rho(\omega)$")
ax.set_xlabel(r"$\omega$")

#%% Normal density vs. By
L_x = 200
L_y = 200
w_0 = 10
Delta = 0.2
mu = -32 + 4*w_0
B_values = np.linspace(0, 3*Delta, 10)
theta = np.pi/2
Lambda = 0.56#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
h = 1e-2
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
omega_values = np.linspace(-6*mu, 6*mu, 10000)
eta = 0.01

n_B_y = np.zeros(len(B_values))
n = np.zeros(len(B_values))
for i, B in enumerate(B_values):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    n_B_y[i] = get_normal_density(omega_values, eta, L_x, L_y, w_0, mu, B_x, B_y, Lambda)

fig, ax = plt.subplots()
ax.plot(B_values, n_B_y, "-o")
ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             +f"; B={B:.2}" + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$n(B_y)$")
plt.tight_layout()
