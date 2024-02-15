#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:17:00 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x

def G_k(k, w_0, Gamma, B_x, B_y, Delta):
    r""" Frozen Green's function.
    
    .. math ::
        w_k(t=0)=2 w_0 \cos(k)
        
        G^f_{k}(t=0,\omega)=\left[\omega - \varepsilon_0-w_{k}( t=0) +i {\Gamma} \right]^{-1}
    
    """
    H_k = (2*w_0*np.cos(k)*np.kron(tau_z, sigma_0) -
           B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)+
           Delta*np.kron(tau_x, sigma_0))
    return np.linalg.inv(-H_k + 1j*Gamma*np.kron(tau_0, sigma_0))

def Rho_k(k, w_0, Gamma, B_x, B_y, Delta):
    r"""
    .. math ::
        \rho^f_k(\theta,\omega) = G_k^f(\theta,\omega) \Gamma [G_k^f(\theta,\omega)]^\dagger
    """
    return G_k(k, w_0, Gamma, B_x, B_y, Delta)@Gamma*np.kron(tau_0, sigma_0)@G_k(k, w_0, Gamma, B_x, B_y, Delta).conj().T

def get_sigma(k, w_0, Gamma, B_x, B_y, Delta):
    r""" DC-superconductivity
    .. math ::
        \sigma=
        -\frac{2}{\pi} w_0^2 \sum_{k=-\pi}^{\pi}   \sin^2 (k) \left[  \rho^f_k(0,0)\right]^2
    """
    sigma_partial = []
    for k_value in k:
        v_k = -2*w_0*np.sin(k_value)*np.kron(tau_0, sigma_0)
        sigma_k = 1/(8*np.pi)*v_k @ Rho_k(k, w_0, Gamma, B_x, B_y, Delta)**2 @ v_k
        sigma_partial.append(sigma_k)
    return np.sum(sigma_partial)