# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:50:55 2024

@author: gabri
"""
import sympy as sp
import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import matplotlib.pyplot as plt

def get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda):
    """ Periodic Hamiltonian in x and y with flux.
    """
    H = (
        -2*w_0*((sp.cos(k_x)*sp.cos(phi_x) + sp.cos(k_y)*sp.cos(phi_y))
               * np.kron(tau_z, sigma_0)
               - (sp.sin(k_x)*sp.sin(phi_x) + sp.sin(k_y)*sp.sin(phi_y))
               * np.kron(tau_0, sigma_0)) - mu * np.kron(tau_z, sigma_0)
        + 2*Lambda*(sp.sin(k_x)*sp.cos(phi_x) * np.kron(tau_z, sigma_y)
                    + sp.cos(k_x)*sp.sin(phi_x) * np.kron(tau_0, sigma_y)
                    - sp.sin(k_y)*sp.cos(phi_y) * np.kron(tau_z, sigma_x)
                    - sp.cos(k_y)*sp.sin(phi_y) * np.kron(tau_0, sigma_x))
        - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
        + Delta*np.kron(tau_x, sigma_0)
            )
    return H

w_0, mu, Delta = sp.symbols("w_0 mu Delta", real=True)
k_x, k_y = sp.symbols("k_(x:y)", real=True)
phi_x, phi_y = sp.symbols("phi_(x:y)", real=True)
B_x, B_y, B_z = sp.symbols("B_(x:z)", real=True)
Lambda = sp.symbols("lambda", real=True)

H = sp.Matrix(get_Hamiltonian(k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda))

#%%

E_k = list(H.eigenvals().keys())
np.save("E_k", E_k)

#%%
E_k = np.load("E_k.npy", allow_pickle=True)
[E_0, E_1, E_2, E_3] =  E_k
E_0 = sp.lambdify([k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda], E_k[0])
E_1 = sp.lambdify([k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda], E_k[1])
E_2 = sp.lambdify([k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda], E_k[2])
E_3 = sp.lambdify([k_x, k_y, phi_x, phi_y, w_0, mu, Delta, B_x, B_y, Lambda], E_k[3])

#%% Plot energy

k_x_values = np.linspace(-np.pi, np.pi, 1000)
fig, ax = plt.subplots()
ax.plot(k_x_values, [E_0(k_x, 0, 0, 0, 10, -32, 0.2,
                         0, 0, 0.56) for k_x in k_x_values])
ax.plot(k_x_values, [E_1(k_x, 0, 0, 0, 10, -32, 0.2,
                         0, 0, 0.56) for k_x in k_x_values])
ax.plot(k_x_values, [E_2(k_x, 0, 0, 0, 10, -32, 0.2,
                         0, 0, 0.56) for k_x in k_x_values])
ax.plot(k_x_values, [E_3(k_x, 0, 0, 0, 10, -32, 0.2,
                         0, 0, 0.56) for k_x in k_x_values])
