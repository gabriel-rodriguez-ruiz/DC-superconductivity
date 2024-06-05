#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:40:16 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
import matplotlib.pyplot as plt
import multiprocessing
from pathlib import Path

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

if __name__ == "__main__":
    L_x = 500
    L_y = 500
    w_0 = 10
    Delta = 0.2
    mu = -40
    B_values = np.linspace(0, 3*Delta, 10)
    theta = np.pi/2
    Lambda = 0.56#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
    omega_values = np.linspace(-6*w_0, 6*w_0, 10000)
    eta = 0.01
    n_cores = 10
    
    params = {"L_x": L_x, "L_y": L_y, "w_0": w_0,
              "mu": mu, "Delta": Delta, "theta": theta,
              "B_values": B_values, "Lambda": Lambda,
              }
    def integrate(B):
        B_x = B * np.cos(theta)
        B_y = B * np.sin(theta)
        n = get_normal_density(omega_values, mu, eta, L_x, L_y, w_0, B_x, B_y, Lambda)
        return n
    
    B_values = np.linspace(0, 3*Delta, 10)
    with multiprocessing.Pool(n_cores) as pool:
        results_pooled = pool.map(integrate, B_values)
    n_B_y = np.array(results_pooled)
    
    data_folder = Path("Data/")
    name = f"normal_n_By_mu_{mu}_L={L_x}_B_y_in_({np.min(np.round(B_values,2))}-{np.max(np.round(B_values,2))})_Delta={Delta}.npz"
    file_to_open = data_folder / name
    np.savez(file_to_open , n_B_y=n_B_y,
             **params)