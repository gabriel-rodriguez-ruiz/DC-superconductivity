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
            -2*w_0*((np.cos(k_x)*np.cos(phi_x) + np.cos(k_y)*np.cos(phi_y))
                   * np.kron(tau_z, sigma_0)
                   - (np.sin(k_x)*np.sin(phi_x) + np.sin(k_y)*np.sin(phi_y))
                   * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
            + 2*Lambda*(np.sin(k_x)*np.cos(phi_x) * np.kron(tau_z, sigma_y)
                        + np.cos(k_x)*np.sin(phi_x) * np.kron(tau_0, sigma_y)
                        - np.sin(k_y)*np.cos(phi_y) * np.kron(tau_z, sigma_x)
                        - np.cos(k_y)*np.sin(phi_y) * np.kron(tau_0, sigma_x))
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

def get_superconducting_density(L_x, L_y, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda, h):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    E = get_Energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda)
    negative_energy = np.where(E<0, E, 0)
    fundamental_energy = 1/2*np.sum(negative_energy, axis=(0, 1, 4))
    n_s = 1/(L_x*L_y) * 2*( fundamental_energy[1] - fundamental_energy[0]) / (h**2)
    return n_s[0]    

#%% Superconducting density vs. 1/(L_x*L_y)

L_values = np.linspace(10, 300, 10, dtype=int)
w_0 = 10
Delta = 0.2
mu = 2*(20*Delta-2*w_0)
theta = np.pi/2
B = 3*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
h = 1e-4
phi_x_values = [0, h]
# phi_x_values = np.linspace(-1/10000, 1/10000, 10)
phi_y_values = [0]    #np.linspace(0, np.pi, 1)

params = {"L_values":L_values, "w_0":w_0, "Delta":Delta,
          "mu":mu, "phi_x_values":phi_x_values,
          "phi_y_values":phi_x_values,
          "B":B,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda
          }

n_L = np.zeros(len(L_values))
for i, L in enumerate(L_values):
    n_L[i] = (get_superconducting_density(L, L, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda, h)
              )
    print(i)
    
fig, ax = plt.subplots()
ax.plot(L_values, n_L, "o")
# ax.plot(1/(L_values)**2, n_L, "o")
# ax.set_xlabel(r"$\frac{1}{L_x L_y}$")
ax.set_xlabel("L")
# ax.set_ylabel(r"$\frac{(n_s)_{xx}}{(n)_{xx}}$")
ax.set_ylabel(r"$n_s$")

ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             +f"; B={B}" + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}")
# ax.set_yscale("log")
plt.tight_layout()
#%%
np.savez("Large_L_limit", L=L, n_L=n_L, Lambda=Lambda,
         Delta=Delta, B=B, theta=theta, mu=mu, w_0=w_0)

#%% Angular dependence

L_x = 50
L_y = 50
w_0 = -10
mu = 0
Delta = 0.1
theta_values = np.linspace(0, np.pi/2, 10)
B = 0.05
Lambda = 0.1
phi_x_values = np.linspace(-np.pi/10000, np.pi/10000, 10)
phi_y_values = [0]    #np.linspace(0, np.pi, 1)
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)

n_theta = np.zeros_like(theta_values)
for i, theta in enumerate(theta_values):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    n_theta[i] = get_superconducting_density(L_x, L_y, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda)
    print(i)
    
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_values, n_theta, "o")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$n_{\theta}$")
plt.tight_layout()

#%% Load data

data = np.load("Large_L_limit")

#%% n_s vs. B_y

L_x = 150
L_y = 150
w_0 = 10
Delta = 0.2
mu = 2*(20*Delta-2*w_0)
theta = np.pi/2
B_values = np.linspace(0, 0.5*Delta, 2)
Lambda = 5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
h = 1e-4
phi_x_values = [0, h]
phi_y_values = [0]
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)

n_B_y = np.zeros(len(B_values))
for i, B in enumerate(B_values):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    n_B_y[i] = get_superconducting_density(L_x, L_y, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda, h)
    print(i)

fig, ax = plt.subplots()
ax.plot(B_values/Delta, n_B_y, "o")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$n(B_y)$")
plt.tight_layout()

#%% Plot energy bands
L_x = 1000
L_y = 1
w_0 = 10
Delta = 0.2
mu = -40#2*(20*Delta-2*w_0)#0
k_F = np.sqrt((4*w_0 + mu)/(w_0))
theta = np.pi/2
B = 3*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.1#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2 #5*Delta/k_F
phi_x_values = [0]
phi_y_values = [0]    #np.linspace(0, np.pi, 1)
k_x_values = np.pi/L_x*np.arange(-L_x, L_x)
k_y_values = [0]#np.pi/L_y*np.arange(-L_y, L_y)

params = {"L_x":L_x, "L_y":L_y, "w_0":w_0, "Delta":Delta,
          "mu":mu, "phi_x_values":phi_x_values,
          "phi_x_values":phi_x_values,
          "phi_y_values":phi_y_values,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda
          }

E = get_Energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda)

fig, ax = plt.subplots()
# ax.plot(k_x_values, E[:,0,0,0,0]/Delta)
# ax.plot(k_x_values, E[:,0,0,0,1]/Delta)
# ax.plot(k_x_values, E[:,0,0,0,2]/Delta)
# ax.plot(k_x_values, E[:,0,0,0,3]/Delta)
ax.plot(k_x_values, E[:,0,0,0,0]/Delta)
ax.plot(k_x_values, E[:,0,0,0,1]/Delta)
ax.plot(k_x_values, E[:,0,0,0,2]/Delta)
ax.plot(k_x_values, E[:,0,0,0,3]/Delta)

ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$\frac{E(k_x)}{\Delta}$")
ax.set_title(f"w_0={w_0}; Delta={Delta}; mu={mu}; Lambda={Lambda:.2}; k_y=0; Bx={B_x:.2}; By={B_y:.2}")
plt.tight_layout()