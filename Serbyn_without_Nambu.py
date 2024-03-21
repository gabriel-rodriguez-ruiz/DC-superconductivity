#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:30:47 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y
import scipy
import matplotlib.pyplot as plt

def G_k(k, epsilon_n, w_0, Gamma, mu):
    r""" Matsubara Green's function.
    
    .. math ::
        G(k, i\epsilon_n)=\left[(i\epsilon_n+i\Gamma)\tau_0\sigma_0-\left(2w_0cos(k)-\mu\right)\tau_z\sigma_0+\Delta\tau_y\sigma_y\right]^{-1}        
        
        \epsilon_n=2\pi/\beta\left(n+\frac{1}{2}\right)
    """
    h_0 = 2*w_0*np.cos(k)-mu
    return 1/(1j*epsilon_n+1j*np.sign(epsilon_n)*Gamma-h_0)

def G_frozen(k, omega, w_0, Gamma, mu):
    h_0 = 2*w_0*np.cos(k)-mu
    return 1/(omega+1j*Gamma-h_0)    

def get_Q_k(k, w_0, Gamma, mu, N, beta, Omega):
    epsilon_n = np.pi/beta*(2*np.arange(-N, N) + 1)
    sumand = np.zeros_like(epsilon_n, dtype=complex)
    for i in range(len(epsilon_n)):
        v = -2*w_0*np.sin(k)
        sumand[i] = (v**2)*G_k(k, epsilon_n[i]+Omega, w_0, Gamma, mu)*G_k(k, epsilon_n[i], w_0, Gamma, mu)
    return -1/(beta)*np.sum(sumand, dtype=complex)

def Rho_k(k, omega, w_0, Gamma, mu):
    h_0 = 2*w_0*np.cos(k)-mu
    return 2*Gamma/((omega+1j*Gamma-h_0)*(omega-1j*Gamma-h_0))

def G_k_Matsubara(k, epsilon_n, omega, w_0, Gamma, mu):
    r""" Matsubara Green's function.
    
    .. math ::
        G(k, i\epsilon_n)=\left[(i\epsilon_n+i\Gamma)\tau_0\sigma_0-\left(2w_0cos(k)-\mu\right)\tau_z\sigma_0+\Delta\tau_y\sigma_y\right]^{-1}        
        
        \epsilon_n=2\pi/\beta\left(n+\frac{1}{2}\right)
    """
    integrand_array = np.zeros_like(omega, dtype=complex)
    for i in range(len(omega)):
        integrand_array[i] = 1/(2*np.pi)*Rho_k(k, omega[i], w_0, Gamma, mu)/(1j*epsilon_n-omega[i])
    integral = np.trapz(integrand_array, omega)
    return integral

def Fermi_function_derivative(omega, beta):
    return-beta*np.exp(beta*omega)/(1 + np.exp(beta*omega))**2

def integrand(k, omega, w_0, Gamma, mu, beta):
    v_k = -2*w_0*np.sin(k)
    return Fermi_function_derivative(omega, beta)*v_k**2*Rho_k(k, omega, w_0, Gamma, mu)**2
def get_sigma_quad_k(k, w_0, Gamma, mu, beta):
    integral = scipy.integrate.quad_vec(lambda omega: integrand(k, omega, w_0, Gamma, mu, beta), -2, 2)[0]
    sigma_k = -1/(4*np.pi)*integral
    return sigma_k
def get_sigma_k(k, omega, w_0, Gamma, mu, beta):
    integrand_array = np.zeros_like(omega, dtype=complex)
    for i in range(len(omega)):
        integrand_array[i] = integrand(k, omega[i], w_0, Gamma, mu, beta)
    integral = np.trapz(integrand_array, omega, axis=0)
    sigma_k = -1/(4*np.pi)*integral
    return sigma_k
def Fermi_function(omega, beta):
    return 1/(1+np.exp(beta*omega))
def get_zero_order_k(k, omega, w_0, Gamma, mu, beta):
    integrand = np.zeros_like(omega)
    for i in range(len(omega)):
        integrand[i] = -2*w_0*np.sin(k)*Fermi_function(omega[i], beta)*Rho_k(k, omega[i], w_0, Gamma, mu)
    integral = np.trapz(integrand, omega)
    return 1/(2*np.pi)*integral

def get_Q_k_Matsubara(k, omega, w_0, Gamma, mu, N, beta):
    epsilon_n = np.pi/beta*(2*np.arange(-N, N) + 1)
    sumand = np.zeros_like(epsilon_n, dtype=complex)
    for i in range(len(epsilon_n)):
        v = -2*w_0*np.sin(k)
        sumand[i] = v**2*G_k_Matsubara(k, epsilon_n[i], omega, w_0, Gamma, mu)**2
    return -1/(2*beta)*np.sum(sumand, dtype=complex)

def get_diamagnetic_term(k, w_0, beta):
    sumand = np.zeros_like(k)
    for i in range(len(k)):
        sumand[i] = -2*w_0*np.cos(k[i])*Fermi_function(2*np.cos(k[i]), beta)
    return np.sum(sumand)

def get_diamagnetic_term_k(k, w_0, beta):
    return -2*w_0*np.cos(k)*Fermi_function(2*np.cos(k), beta)
#%%

beta = 10
N = 100000
Gamma = 0.01
mu = 0
w_0 = 1
omega = np.linspace(-1, 1, 1000)
epsilon_n = np.pi/beta*(2*np.arange(-N, N) + 1)
Omega = 2*np.pi/beta*(1)    #bosonic frequency
# k = np.pi/2
# Q_k = get_Q_k(k, w_0, Gamma, mu, N, beta)
# sigma_quad_k = get_sigma_quad_k(k, w_0, Gamma, mu, beta)
# sigma_k = get_sigma_k(k, omega, w_0, Gamma, mu, beta)

L = 200
k = 2*np.pi/L*np.arange(0, L)
Q_k = [1/L*(get_Q_k(k, w_0, Gamma, mu, N, beta, Omega)) for k in k]
# Q_k_Matsubara = [get_Q_k_Matsubara(k, omega, w_0, Gamma, mu, N, beta) for k in k]
sigma_k = [1/L*get_sigma_k(k, omega, w_0, Gamma, mu, beta) for k in k]
# sigma_quad_k = [get_sigma_quad_k(k, w_0, Gamma, mu, beta) for k in k]
# zero_order_k = [get_zero_order_k(k, omega, w_0, Gamma, mu, beta) for k in k]
diamagnetic_term = get_diamagnetic_term(k, w_0, beta)

fig, ax = plt.subplots()
# ax.plot(k, Q_k, label="Q")
ax.plot(k, sigma_k, label=r"$\sigma$")
ax.plot(k, 1/Omega*np.array(Q_k), label="Q/Gamma")
ax.plot(k, diamagnetic_term, label="Diamagnetic")
ax.set_xlabel("k")  
plt.legend()

#%%
beta = 10
N = 100000
Gamma = 0.1
mu = 0
w_0 = 1
omega = np.linspace(-1, 1, 1000)
epsilon_n = np.pi/beta*(2*np.arange(-N, N) + 1)
Omega = np.pi/beta*(2*np.arange(1, 10))     #boson frequency

L = 200
k = np.pi/4
Q_k = [1/L*(get_Q_k(k, w_0, Gamma, mu, N, beta, Omega)) for Omega in Omega]
sigma_k = 1/L*get_sigma_k(k, omega, w_0, Gamma, mu, beta)
fig, ax = plt.subplots()
ax.plot(0, sigma_k, "o", label=r"$\sigma$")
ax.plot(Omega, np.array(Q_k), "o",label="Q/Gamma")
ax.set_xlabel(r"$\Omega$")  
plt.legend()
