#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:17:00 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_x
import scipy.integrate

def G_k(k, omega, w_0, Gamma, B_x, B_y, Delta, mu):
    r""" Frozen Green's function.
    
    .. math ::
        w_k(t=0)= (2 w_0 \cos(k) - \mu ) \tau_z\sigma_0
        
        G^f_{k}(t=0,\omega)=\left[w\tau_0\sigma_0-(w_{k}( t=0) -B_x\tau_0\sigma_x-B_y\tau_0\sigma_y +\Delta\tau_x\sigma_0)+i \Gamma\tau_0\sigma_0 \right]^{-1}
    
    """
    H_k = ((2*w_0*np.cos(k)-mu)*np.kron(tau_z, sigma_0) -
           B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)+
           Delta*np.kron(tau_x, sigma_0))
    return np.linalg.inv(omega*np.kron(tau_0, sigma_0)-H_k + 1j*Gamma*np.kron(tau_0, sigma_0))

def Rho_k(k, omega, w_0, Gamma, B_x, B_y, Delta, mu):
    r"""
    .. math ::
        \rho^f_k(\theta=0,\omega) = G_k^f(\theta=0,\omega) \Gamma [G_k^f(\theta=0,\omega)]^\dagger
    """
    return G_k(k, omega, w_0, Gamma, B_x, B_y, Delta, mu)@(Gamma*np.kron(tau_0, sigma_0))@(G_k(k, omega, w_0, Gamma, B_x, B_y, Delta, mu).conj().T)

def Fermi_function_derivative(omega, beta):
    return-beta*np.exp(beta*omega)/(1 + np.exp(beta*omega))**2

def integrand(k, omega, w_0, Gamma, B_x, B_y, Delta, mu, beta):
    return Fermi_function_derivative(omega, beta)*Rho_k(k, omega, w_0, Gamma, B_x, B_y, Delta, mu)@Rho_k(k, omega, w_0, Gamma, B_x, B_y, Delta, mu)

def get_sigma(k, omega, w_0, Gamma, B_x, B_y, Delta, mu, beta):
    r""" DC-superconductivity
    
    .. math ::
        \sigma=\frac{1}{8\pi} Tr\left[\sum_{k=-\pi}^{\pi} v(k) \rho^f_k(\theta=0,\omega=0)^2 v(k) \right]
    """
    sigma_partial = []
    for k_value in k:
        integrand = []
        for omega_value in omega:
            integrand.append(Fermi_function_derivative(omega_value, beta)*Rho_k(k_value, omega_value, w_0, Gamma, B_x, B_y, Delta, mu)@Rho_k(k_value, omega_value, w_0, Gamma, B_x, B_y, Delta, mu))
        integral = np.trapz(np.array(integrand), x=omega, axis=0)
        v_k = -2*w_0*np.sin(k_value)*np.kron(tau_0, sigma_0)
        sigma_k = -1/(8*np.pi)*v_k @ integral @ v_k
        sigma_partial.append(sigma_k)
    return np.trace(np.sum(sigma_partial, axis=0))

def get_sigma_quad(k, w_0, Gamma, B_x, B_y, Delta, mu, beta):
    r""" DC-superconductivity
    
    .. math ::
        \sigma=-\frac{1}{4} \sum_{k=-\pi}^{\pi}\int \frac{d\omega}{2\pi} v(k)\frac{\partial f}{\partial \omega}\rho(0,\omega)^2 v(k)    """
    sigma_partial = []
    for k_value in k:
        integral = scipy.integrate.quad_vec(lambda omega: integrand(k_value, omega, w_0, Gamma, B_x, B_y, Delta, mu, beta), -1, 1)[0]
        v_k = -2*w_0*np.sin(k_value)*np.kron(tau_0, sigma_0)
        sigma_k = -1/(8*np.pi)*v_k @ integral @ v_k
        sigma_partial.append(sigma_k)
    return np.trace(np.sum(sigma_partial, axis=0))