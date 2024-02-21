#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:20:04 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y
import scipy.integrate as integrate

def G_k(k, epsilon_n, w_0, Gamma, Delta, mu):
    r""" Matsubara Green's function.
    
    .. math ::
        G(k, i\epsilon_n)=\left[(i\epsilon_n+i\Gamma)\tau_0\sigma_0-\left(2w_0cos(k)-\mu\right)\tau_z\sigma_0+\Delta\tau_y\sigma_y\right]^{-1}        
        
        \epsilon_n=2\pi/\beta\left(n+\frac{1}{2}\right)
    """
    h_0 = (2*w_0*np.cos(k)-mu)*np.kron(tau_z, sigma_0) - Delta*np.kron(tau_y, sigma_y)
    return np.linalg.inv((1j*epsilon_n+1j*Gamma)*np.kron(tau_0, sigma_0)-h_0)

def integrand(k, epsilon_n, w_0, Gamma, Delta, mu):
    r"""
    
    .. math ::
        Tr\left[ \frac{k^2}{m^2} G(k,i\epsilon_n)G(k,i\epsilon_n)\right]]    
    """
    v = -2*w_0*np.sin(k)*np.kron(tau_0, sigma_0)
    return np.trace(v@v@G_k(k, epsilon_n, w_0, Gamma, Delta, mu)@G_k(k, epsilon_n, w_0, Gamma, Delta, mu))

def get_Q(k, w_0, Gamma, Delta, mu, N, beta):
    r"""Kernel
    
    .. math ::
        Q(0,0)=-e^2 \frac{1}{2\beta} \sum_{\epsilon_n}\frac{1}{L}\sum_k Tr\left[ v^2 G(k,i\epsilon_n)G(k,i\epsilon_n)\right]
    """
    epsilon_n = 2*np.pi/beta*(np.arange(N)+1/2)
    dk = np.diff(k)[0]
    sumand = []
    for E in epsilon_n:
        sumand_k = []
        for k_value in k:
            v = -2*w_0*np.sin(k_value)*np.kron(tau_0, sigma_0)
            sumand_k.append(v@v@G_k(k_value, E, w_0, Gamma, Delta, mu)@G_k(k_value, E, w_0, Gamma, Delta, mu))
        sumand.append(dk/(2*np.pi)*np.sum(sumand_k, axis=0))
    sumand = np.array(sumand)
    return -1/(4*np.pi*beta)*np.sum(sumand)
        