#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:28:23 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
import matplotlib.pyplot as plt

def get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda):
    """ Periodic Hamiltonian in x and y with flux.
    """
    return (
            2*w_0*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
                   * np.kron(tau_z, sigma_0)
                   - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
                   * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
            + 2*Lambda*(np.sin(k_x)*np.cos(phi_x) * np.kron(tau_0, sigma_y)
                        + np.cos(k_x)*np.sin(phi_x) * np.kron(tau_z, sigma_y)
                        - np.sin(k_y)*np.cos(phi_y) * np.kron(tau_0, sigma_x)
                        - np.cos(k_y)*np.sin(phi_y) * np.kron(tau_z, sigma_x))
            - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
            + Delta*np.kron(tau_x, sigma_0)
            ) * 1/2

def get_Energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda):
    """
    """
    energies = np.zeros((len(k_x_values), len(k_y_values),
                        len(phi_x_values), len(phi_y_values), 4))
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for k, phi_x in enumerate(phi_x_values):
                for l, phi_y in enumerate(phi_y_values):
                    for m in range(4):
                        H = get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda)
                        energies[i, j, k, l, m] = np.linalg.eigvalsh(H)[m]
    return energies

def get_superconducting_density(L_x, L_y, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    E = get_Energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda)
    negative_energy = np.where(E<0, E, 0)
    fundamental_energy = 1/2*np.sum(negative_energy, axis=(0, 1, 3, 4))
    current = np.gradient(fundamental_energy, phi_x_values)
    D_s = np.gradient(current, phi_x_values)
    n_s = 1/(L_x*L_y) * D_s[len(phi_x_values)//2]
    return n_s

#%%
L_x = 10
L_y = 10
w_0 = 1
mu = 0
Delta = 0.1
theta = np.pi/2
B = 1
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.1
phi_x_values = np.linspace(-np.pi/10000, np.pi/10000, 10)
phi_y_values = [0]    #np.linspace(0, np.pi, 1)
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)

params = {"L_x":L_x, "L_y":L_y, "w_0":w_0, "Delta":Delta,
          "mu":mu, "phi_x_values":phi_x_values,
          "phi_x_values":phi_x_values,
          "phi_y_values":phi_y_values,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda
          }

E = get_Energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda)


#%% Energy vs. flux
# negative_energy = np.where(E_k_x_phi_x<0, E_k_x_phi_x, 0)
negative_energy = np.where(E<0, E, 0)

fig, ax = plt.subplots()
for i, k_x in enumerate(k_x_values):
    for j, k_y in enumerate(k_y_values):
        ax.plot(phi_x_values, E[i,j,:,:,0], "b")
        ax.plot(phi_x_values, E[i,j,:,:,1], "b")
        ax.plot(phi_x_values, E[i,j,:,:,2], "b")
        ax.plot(phi_x_values, E[i,j,:,:,3], "b")
        ax.plot(phi_x_values, negative_energy[i,j,:,:,0], "r")
        ax.plot(phi_x_values, negative_energy[i,j,:,:,1], "r")
        ax.plot(phi_x_values, negative_energy[i,j,:,:,2], "r")
        ax.plot(phi_x_values, negative_energy[i,j,:,:,3], "r")

ax.set_xlabel(r"$\phi_x$")
ax.set_ylabel(r"$E_{n}(\phi_x)$")

#%% Fundamental energy

fundamental_energy = 1/2*np.sum(negative_energy, axis=(0, 1, 3, 4))

fig, ax = plt.subplots()
ax.plot(phi_x_values, fundamental_energy, "o-")
ax.set_xlabel(r"$\phi_x$")
ax.set_ylabel(r"$E_{0}(\phi_x)$")

#%% Current (First derivative)
current = np.gradient(fundamental_energy, phi_x_values)

fig, ax = plt.subplots()
ax.plot(phi_x_values, current, "o-")
ax.set_xlabel(r"$\phi_x$")
ax.set_ylabel(r"$\frac{\partial E_{0}}{\partial \phi_x}(\phi_x)$")

#%% Second derivative

D_s = np.gradient(current, phi_x_values)

fig, ax = plt.subplots()
ax.plot(phi_x_values, D_s, "o-")
ax.set_xlabel(r"$\phi_x$")
ax.set_ylabel(r"$\frac{\partial^2 E_{0}}{\partial \phi_x^2}(\phi_x)$")

n_s = 1/(L_x*L_y) * D_s[len(phi_x_values)//2]

#%% Superconducting density vs. 1/(L_x*L_y)

L_values = np.linspace(10, 200, 10, dtype=int)
w_0 = 1
mu = 0
Delta = 0.1
theta = np.pi/2
B = 1
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.1
phi_x_values = np.linspace(-1/10000, 1/10000, 10)
phi_y_values = [0]    #np.linspace(0, np.pi, 1)

n_L = np.zeros(len(L_values))
for i, L in enumerate(L_values):
    n_L[i] = get_superconducting_density(L, L, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda)
    print(i)
    
fig, ax = plt.subplots()
ax.plot(L_values, n_L, "o")
# ax.plot(1/(L_values)**2, n_L, "o")
# ax.set_xlabel(r"$\frac{1}{L_x L_y}$")
ax.set_xlabel("L")
ax.set_ylabel(r"$(n_s)_{xx}$")
ax.set_title(r"$\lambda=$" + f"{Lambda}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             +f"; B={B}")
# ax.set_yscale("log")
plt.tight_layout()
#%%
np.savez("Large_L_limit", L=L, n_L=n_L, Lambda=Lambda,
         Delta=Delta, B=B, theta=theta)

#%% Angular dependence

L_x = 50
L_y = 50
w_0 = 1
mu = 0
Delta = 0.1
theta_values = np.linspace(0, np.pi/2, 10)
Lambda = 0.1
phi_x_values = np.linspace(-np.pi/10000, np.pi/10000, 10)
phi_y_values = [0]    #np.linspace(0, np.pi, 1)
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)

n_theta = np.zeros_like(theta_values)
for i, theta in enumerate(theta_values):
    B_x = np.cos(theta)
    B_y = np.sin(theta)
    n_theta[i] = get_superconducting_density(L_x, L_y, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda)
    print(i)
    
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_values, n_theta, "o")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$n_{\theta}$")
plt.tight_layout()