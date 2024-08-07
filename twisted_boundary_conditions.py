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
    H = (
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
    return H

def get_Energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda):
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

def get_superconducting_density(L_x, L_y, w_0, mu, Delta, B_x, B_y, Lambda, h):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    phi_x_values = [-h, 0, h]
    phi_y_values = [-h, 0, h]
    E = get_Energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda)
    negative_energy = np.where(E<0, E, 0)
    fundamental_energy = 1/2*np.sum(negative_energy, axis=(0, 1, 4))
    n_s_xx = 1/w_0 * 1/(L_x*L_y) * ( fundamental_energy[2,1] - 2*fundamental_energy[1,1] + fundamental_energy[0,1]) / h**2
    n_s_yy = 1/w_0 * 1/(L_x*L_y) * ( fundamental_energy[1,2] - 2*fundamental_energy[1,1] + fundamental_energy[1,0]) / h**2
    n_s_xy = 1/w_0 * 1/(L_x*L_y) * ( fundamental_energy[2,2] - fundamental_energy[2,0] - fundamental_energy[0,2] + fundamental_energy[0,0]) / h**2
    return n_s_xx, n_s_yy, n_s_xy

def get_Green_function(omega, k_x_values, k_y_values, w_0, mu, Delta, B_x, B_y, Lambda):
    G = np.zeros((len(k_x_values), len(k_y_values),
                  4, 4), dtype=complex)
    for i, k_x in enumerate(k_x_values):
        for j, k_y in enumerate(k_y_values):
            for l in range(4):
                for m in range(4):
                    G[i, j, l, m] = np.linalg.inv(omega*np.kron(tau_0, sigma_0)
                                                  - get_Hamiltonian(k_x, k_y, 0, 0, w_0, mu, Delta, B_x, B_y, Lambda) 
                                                  )[l, m]                
    return G

def get_DOS(omega, eta, L_x, L_y, w_0, mu, Delta, B_x, B_y, Lambda):
    k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
    k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
    G = get_Green_function(omega+1j*eta, k_x_values, k_y_values, w_0, mu, Delta, B_x, B_y, Lambda)
    return 1/(L_x*L_y) * 1/np.pi*np.sum(-np.imag(G), axis=(0,1))

#%% Superconducting density vs. 1/(L_x*L_y)

L_values = np.linspace(10, 100, 10, dtype=int)
w_0 = 10
Delta = 0
mu = -40
theta = np.pi/2
B = 0
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.56
h = 1e-2


params = {"L_values":L_values, "w_0":w_0, "Delta":Delta,
          "mu":mu,
          "B":B,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda
          }

n_L = np.zeros((len(L_values), 3))
for i, L in enumerate(L_values):
    L_x = L
    L_y = L
    n_L[i, 0], n_L[i, 1], n_L[i, 2] = (get_superconducting_density(L_x, L_y, w_0, mu, Delta, B_x, B_y, Lambda, h)
              )
    print(i)
    
fig, ax = plt.subplots()
ax.plot(L_values, n_L[:,0], "o", label=r"$n_{s,\perp}$")
ax.plot(L_values, n_L[:,1], "o", label=r"$n_{s,\parallel}$")
ax.legend()
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

L_x = 150
L_y = 150
w_0 = 10
Delta = 0.2
mu = -40
theta_values = np.linspace(0, np.pi, 20)
B = 0.5*Delta
Lambda = 5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
h = 1e-2
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)
n = 0.136

n_theta = np.zeros((len(theta_values), 3))
for i, theta in enumerate(theta_values):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    n_theta[i,0], n_theta[i,1], n_theta[i,2] = get_superconducting_density(L_x, L_y, w_0, mu, Delta, B_x, B_y, Lambda, h)
    print(i)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_values, n_theta[:,0]/n, "or", label=r"$n_{xx}$")
ax.plot(theta_values, n_theta[:,1]/n, "ob", label=r"$n_{yy}$")
# ax.plot(theta_values, np.sqrt(n_theta[:,0]**2+n_theta[:,1]**2), "og", label=r"$|n_s|^2$")
# ax.plot(theta_values+np.pi/2, n_theta[:,0], "or", label=r"$n_{s,\perp}$")
# ax.plot(theta_values+np.pi/2, n_theta[:,1], "ob", label=r"$n_{s,\parallel}$")
# ax.plot(theta_values+np.pi/2, np.sqrt(n_theta[:,0]**2+n_theta[:,1]**2), "og", label=r"$|n_s|^2$")
# ax.plot(theta_values+np.pi, n_theta[:,0], "or", label=r"$n_{s,\perp}$")
# ax.plot(theta_values+np.pi, n_theta[:,1], "ob", label=r"$n_{s,\parallel}$")
# ax.plot(theta_values+np.pi, np.sqrt(n_theta[:,0]**2+n_theta[:,1]**2), "og", label=r"$|n_s|^2$")
# ax.plot(theta_values+3*np.pi/2, n_theta[:,0], "or", label=r"$n_{s,\perp}$")
# ax.plot(theta_values+3*np.pi/2, n_theta[:,1], "ob", label=r"$n_{s,\parallel}$")
# ax.plot(theta_values+3*np.pi/2, np.sqrt(n_theta[:,0]**2+n_theta[:,1]**2), "og", label=r"$|n_s|^2$")


ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$n_{\theta}$")
ax.legend()
plt.tight_layout()

#%%
np.savez("Data/n_angle_mu=-40_L=400", n_theta=n_theta, theta_values=theta_values,
         B=B)
#%% Load data

data = np.load("Large_L_limit")

#%% n_s vs. B_y

L_x = 400
L_y = 400
w_0 = 10
Delta = 0.2
mu = -32#2*(20*Delta-2*w_0)
theta = np.pi/2
B_values = np.linspace(0, 3*Delta, 20)
Lambda = 0.56#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
h = 1e-2
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)

params = {"L_x": L_x, "L_y": L_y, "w_0": w_0,
          "mu": mu, "Delta": Delta, "theta": theta,
          "B_values": B_values, "Lambda": Lambda,
          "h": h , "k_x_values": k_x_values,
          "k_y_values": k_y_values, "h": h,
          "Lambda": Lambda}

n_B_y = np.zeros((len(B_values), 3))
n = np.zeros(len(B_values))
for i, B in enumerate(B_values):
    B_x = B * np.cos(theta)
    B_y = B * np.sin(theta)
    n_B_y[i,0], n_B_y[i,1], n_B_y[i,2] = get_superconducting_density(L_x, L_y, w_0, mu, Delta, B_x, B_y, Lambda, h)
    print(i)

fig, ax = plt.subplots()
ax.plot(B_values/Delta, n_B_y[:,0], "-o",  label=r"$n_{s,\perp}$")
ax.plot(B_values/Delta, n_B_y[:,1], "-o",  label=r"$n_{s,\parallel}$")
ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
             +f"; B={B:.2}" + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$n(B_y)$")
ax.legend()
plt.tight_layout()

#%%
from pathlib import Path

data_folder = Path("Data/")

file_to_open = data_folder / "n_By_mu_-32_L=400.npz"
# np.savez(file_to_open , n_B_y=n_B_y, **params)

#%% Load data
data_folder = Path("Data/")

file_to_open = data_folder / "n_By_mu_-32_L=400.npz"
data = np.load(file_to_open)
n_B_y = data["n_B_y"]
# Data_n = np.load(data_folder / "n_mu_-40_L=400.npz")  #does not change approx. with magnetic field
# Data_n = np.load(data_folder / "n_mu_-32_L=400.npz")  #does not change approx. with magnetic field
# n = Data_n["n"] #n=0.136
n = n_B_y[0][0]

B_values = data["B_values"]
Delta = data["Delta"]
mu = data["mu"]
Lambda = 0.56
theta = np.pi/2
w_0 = 10

fig, ax = plt.subplots()
ax.plot(B_values/Delta, np.sqrt(n_B_y[:,0]/n), "-o",  label=r"$n_{s,\perp}$")
ax.plot(B_values/Delta, np.sqrt(n_B_y[:,1]/n), "-o",  label=r"$n_{s,\parallel}$")
ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{theta:.3}"
              + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}")
ax.set_xlabel(r"$\frac{B_y}{\Delta}$")
ax.set_ylabel(r"$\sqrt{\frac{n(B_y)}{n}}$")
ax.set_ylim(-0.4, 1.1)
ax.legend()
plt.tight_layout()

#%% Plot energy bands
from matplotlib import cm

L_x = 100
L_y = 100
w_0 = 10
Delta = 0.4#0.2
mu = -39.4#-32
theta = np.pi/4
B = 1#3*Delta
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 0.5#0.56 #5*Delta/k_F
phi_x_values = [0]
phi_y_values = [0]    #np.linspace(0, np.pi, 1)
k_x_values = np.pi/L_x*np.arange(-L_x, L_x)
k_y_values = np.pi/L_y*np.arange(-L_y, L_y)

params = {"L_x":L_x, "L_y":L_y, "w_0":w_0, "Delta":Delta,
          "mu":mu, "phi_x_values":phi_x_values,
          "phi_x_values":phi_x_values,
          "phi_y_values":phi_y_values,
          "B_x":B_x, "B_y":B_y, "Lambda":Lambda
          }

E = get_Energy(k_x_values, k_y_values, phi_x_values, phi_y_values, w_0, mu, Delta, B_x, B_y, Lambda)

fig, ax = plt.subplots()
ax.plot(k_x_values, E[:,L_y,0,0,0])
ax.plot(k_x_values, E[:,L_y,0,0,1])
ax.plot(k_x_values, E[:,L_y,0,0,2])
ax.plot(k_x_values, E[:,L_y,0,0,3])
# ax.plot(k_x_values, E[:,L_y,0,0,0]/Delta)
# ax.plot(k_x_values, E[:,L_y,0,0,1]/Delta)
# ax.plot(k_x_values, E[:,L_y,0,0,2]/Delta)
# ax.plot(k_x_values, E[:,L_y,0,0,3]/Delta)

ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$\frac{E(k_x)}{\Delta}$")
ax.set_title(f"w_0={w_0}; Delta={Delta}; mu={mu}; Lambda={Lambda:.2}; k_y=0; Bx={B_x:.2}; By={B_y:.2}")
plt.tight_layout()

#X, Y = np.meshgrid(k_x_values, k_y_values)
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#Z = np.append(E[:,:,0,0,1], E[:,:,0,0,2],axis=0)
# CS = ax.contour(X, Y, Z)
# ax.clabel(CS, inline=True, fontsize=10)
#surf = ax.plot_surface(np.append(Y, Y, axis=0), np.append(X, X, axis=0) , Z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
# ax.plot_surface(X, Y, E[:,:,0,0,2], cmap='PiYG',
                       # linewidth=0, antialiased=False)
# ax.set_xlabel(r"$k_x$")
# ax.set_ylabel(r"$k_y$")
# ax.set_zlabel(r'$E$')

# fig, ax = plt.subplots()
# C1 = ax.contour(Y, X, E[:,:,0,0,1]>0, 0, colors="orange") #notice the inversion of X and Y
# C2 = ax.contour(Y, X, E[:,:,0,0,2]<0, 0, colors="green")
# ax.clabel(C1, inline=True, fontsize=10)
# ax.clabel(C2, inline=True, fontsize=10)
# ax.set_xlabel(r"$k_x$")
# ax.set_ylabel(r"$k_y$")

#%% Density of states
L_x = 10
L_y = 10
w_0 = 10
Delta = 0.2
mu = 2*(20*Delta-2*w_0)
theta = np.pi/2
B = 0
B_x = B * np.cos(theta)
B_y = B * np.sin(theta)
Lambda = 5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
omega_values = np.linspace(-4*Delta, 4*Delta, 50)
eta = Delta/5

DOS = np.zeros((len(omega_values), 4, 4))
for i, omega in enumerate(omega_values):
    for j in range(4):
        for k in range(4):
            DOS[i, j, k] = get_DOS(omega, eta, L_x, L_y, w_0, mu, Delta, B_x, B_y, Lambda)[j, k]
    print(i)

fig, ax = plt.subplots()
ax.plot(omega_values/Delta, DOS[:, 0, 0])
ax.plot(omega_values/Delta, DOS[:, 1, 1])

ax.set_ylabel(r"$\rho(\omega)$")
ax.set_xlabel(r"$\frac{\omega}{\Delta}$")

#%% n vs. h

L_x = 50
L_y = 50
w_0 = 10
Delta = 0.2
mu = -32#2*(20*Delta-2*w_0)
theta = np.pi/2
B = 2*Delta
Lambda = 0.56#5*Delta/np.sqrt((4*w_0 + mu)/w_0)/2
h_values = np.linspace(1e-4, 1e-2, 10)
k_x_values = 2*np.pi/L_x*np.arange(0, L_x)
k_y_values = 2*np.pi/L_y*np.arange(0, L_y)

n_h = np.zeros((len(h_values), 3))
n = np.zeros(len(h_values))
for i, h in enumerate(h_values):
    n_h[i,0], n_h[i,1], n_h[i,2] = get_superconducting_density(L_x, L_y, w_0, mu, Delta, B_x, B_y, Lambda, h)
    print(i)

fig, ax = plt.subplots()
ax.plot(h_values, n_h[:,0], "-o",  label=r"$n_{s,\perp}$")
ax.plot(h_values, n_h[:,1], "-o",  label=r"$n_{s,\parallel}$")
ax.set_title(r"$\lambda=$" + f"{Lambda:.2}"
              +r"; $\Delta=$" + f"{Delta}"
              +r"; $\theta=$" + f"{theta:.3}"
              +f"; B={B:.2}" + r"; $\mu$"+f"={mu}"
              +r"; $w_0$"+f"={w_0}")
ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$n(h)$")
ax.legend()
plt.tight_layout()