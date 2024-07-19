#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:18:57 2024

@author: gabriel
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder = Path("Data/")

file_path = data_folder / 'resonance_shift_new.dat'

data = pd.read_table(file_path, dtype=float,
                     header=1, sep='\s+',
                      names=["B", r"$\Delta f_{\perp}(3.7 GHz)$",
                             r"$\Delta f_{\parallel}(3.7 GHz)$", r"$\Delta f_{\perp}(4.4 GHz)$",
                             r"$\Delta f_{\parallel}(4.4 GHz)$"])

fig, ax = plt.subplots()
for name in data.columns:
    if name !="B":
        ax.plot(data["B"]**2, data[name], label=name)
        ax.legend()

# fig, ax = plt.subplots()
# data.plot(x=0, ax=ax, marker="o", markersize=2)

ax.set_xlabel(r"$B^2$ [$T^2$]")
ax.set_ylabel(r"$\Delta f [Hz]$")
ax.scatter(data["B"][50]**2, data[r"$\Delta f_{\perp}(3.7 GHz)$"][50])
a, b = np.polyfit(data["B"][:50]**2,
                              data[r"$\Delta f_{\perp}(3.7 GHz)$"][:50]
                              , deg=1)
f = lambda x: a * x + b
ax.plot(data["B"][:100]**2, f(data["B"][:100]**2), "--")
m, n = np.polyfit(data["B"][50:259]**2,
                              data[r"$\Delta f_{\perp}(3.7 GHz)$"][50:259]
                              , deg=1)
h = lambda x: m * x + n
ax.plot(data["B"][50:259]**2, h(data["B"][50:259]**2), "--")

ax.scatter(data["B"][100]**2, data[r"$\Delta f_{\parallel}(3.7 GHz)$"][100])
c, d = np.polyfit(data["B"][:100]**2,
                              data[r"$\Delta f_{\parallel}(3.7 GHz)$"][:100]
                              , deg=1)
g = lambda x: c * x + d
ax.plot(data["B"][:200]**2, g(data["B"][:200]**2), "--")
o, p = np.polyfit(data["B"][100:]**2,
                              data[r"$\Delta f_{\parallel}(3.7 GHz)$"][100:]
                              , deg=1)
j = lambda x: o * x + p
ax.plot(data["B"][100:]**2, j(data["B"][100:]**2), "--")

fig.text(0.3, 0.8, r"$B^*_{\perp}=$"+f"{np.round(data['B'][50],3)}T")
fig.text(0.3, 0.7, r"$B^*_{\parallel}=$"+f"{np.round(data['B'][100],3)}T")

#%%fitting

data_folder = Path("../AC-superconductivity/Data/")
file_to_open = data_folder / "Response_kernel_vs_B_mu=-39_L=100_Gamma=0.1_Omega=0.npz"
Data = np.load(file_to_open)

K = Data["K"]
B_values = Data["B_values"]
Lambda = Data["Lambda"]
Delta = Data["Delta"]
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
part = Data["part"]
Omega = Data["Omega"]
Gamma = Data["Gamma"]
# L_x = Data["L_x"]
# L_y = Data["L_y"]


f_geo = np.array([3.7e9, 4.4e9])
f_kin_perp = np.sqrt(K[:, 0, 0]/K[0, 0, 0])
f_kin_para = np.sqrt(K[:, 1, 0]/K[0, 1, 0])
Delta_f_perp = 1e8 * np.array([f_geo * f_kin / np.sqrt(f_geo**2 + f_kin**2) for f_kin in f_kin_perp])
Delta_f_para = 1e8 * np.array([f_geo * f_kin / np.sqrt(f_geo**2 + f_kin**2) for f_kin in f_kin_para])

fig, ax = plt.subplots()
ax.plot(B_values**2, f_kin_perp, "-o",  label=r"$\perp$")
ax.plot(B_values**2, f_kin_para, "-o",  label=r"$\parallel$")
ax.set_xlabel(r"$B^2$")
ax.set_ylabel(r"$\sqrt{K}$")
ax.legend()
# ax.plot(B_values**2, Delta_f_para[:,1]-1e8, "-o",  label=r"$K^{(L)}_{xx}(\Omega=$"+f"{Omega}"+r"$,\mu=$"+f"{np.round(mu,2)}"+r", $\lambda=$"+f"{Lambda})")
