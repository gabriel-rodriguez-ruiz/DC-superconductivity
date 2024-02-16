#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:17:00 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x

def G_k(k, w_0, Gamma, B_x, B_y, Delta, mu):
    r""" Frozen Green's function.
    
    .. math ::
        w_k(t=0)= (2 w_0 \cos(k) - \mu ) \tau_z\sigma_0
        
        G^f_{k}(t=0,\omega=0)=\left[-(w_{k}( t=0) -B_x\tau_0\sigma_x-B_y\tau_0\sigma_y +\Delta\tau_x\sigma_0)+i \Gamma\tau_0\sigma_0 \right]^{-1}
    
    """
    H_k = ((2*w_0*np.cos(k)-mu)*np.kron(tau_z, sigma_0) -
           B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)+
           Delta*np.kron(tau_x, sigma_0))
    return np.linalg.inv(-H_k + 1j*Gamma*np.kron(tau_0, sigma_0))

def Rho_k(k, w_0, Gamma, B_x, B_y, Delta, mu):
    r"""
    .. math ::
        \rho^f_k(\theta,\omega) = G_k^f(\theta,\omega) \Gamma [G_k^f(\theta,\omega)]^\dagger
    """
    return G_k(k, w_0, Gamma, B_x, B_y, Delta, mu)@(Gamma*np.kron(tau_0, sigma_0))@(G_k(k, w_0, Gamma, B_x, B_y, Delta, mu).conj().T)

def get_sigma(k, w_0, Gamma, B_x, B_y, Delta, mu):
    r""" DC-superconductivity
    
    .. math ::
        \sigma=\frac{1}{8\pi} Tr\left[\sum_{k=-\pi}^{\pi} v(k) \rho^f_k(\theta=0,\omega=0)^2 v(k) \right]
    """
    sigma_partial = []
    for k_value in k:
        v_k = -2*w_0*np.sin(k_value)*np.kron(tau_0, sigma_0)
        sigma_k = 1/(8*np.pi)*v_k @ Rho_k(k_value, w_0, Gamma, B_x, B_y, Delta, mu) @ Rho_k(k_value, w_0, Gamma, B_x, B_y, Delta, mu) @ v_k
        sigma_partial.append(sigma_k)
    return np.trace(np.sum(sigma_partial, axis=0))