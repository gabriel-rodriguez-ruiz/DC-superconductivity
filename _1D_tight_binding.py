#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:11:55 2024

@author: gabriel
"""

import numpy as np

def G_k(k, omega, E_0, w_0, Gamma):
    r""" Frozen Green's function.
    
    .. math ::
        w_k(t=0)=2 w_0 \cos(k)
        
        G^f_{k}(t=0,\omega)=\left[\omega - \varepsilon_0-w_{k}( t=0) +i {\Gamma} \right]^{-1}
    
    """
    w_k = 2*w_0*np.cos(k)
    return (omega - E_0 - w_k + 1j*Gamma)**(-1)

def Rho_k(k, omega, E_0, w_0, Gamma):
    r"""
    .. math ::
        \rho^f_k(\theta,\omega) = G_k^f(\theta,\omega) \Gamma [G_k^f(\theta,\omega)]^\dagger
    """
    return G_k(k, omega, E_0, w_0, Gamma)*np.conj(G_k(k, omega, E_0, w_0, Gamma))*Gamma

def get_sigma(k, omega, E_0, w_0, Gamma):
    r""" DC-superconductivity
    .. math ::
        \sigma=
        -\frac{2}{\pi} w_0^2 \sum_{k=-\pi}^{\pi}   \sin^2 (k) \left[  \rho^f_k(0,0)\right]^2
    """
    sigma_partial = []
    for k_value in k:
        sigma_k = -2/np.pi*w_0**2 * np.sin(k_value)**2 * Rho_k(k_value, omega, E_0, w_0, Gamma)**2
        sigma_partial.append(sigma_k)
    return np.sum(sigma_partial)