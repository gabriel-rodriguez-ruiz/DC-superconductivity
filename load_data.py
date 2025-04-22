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

#%%
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#%%

data_folder = Path("Data/")

file_path = data_folder / 'n_s_4_9GHz.dat'

data = pd.read_table(file_path, dtype=float,
                     header=0, sep='\t\t',
                     names=["field 0°", "n_s 0°", "n_s_error 0°", "field 90°",
                            "n_s 90°", "n_s_error 90°"],
                     engine='python')

fig, ax = plt.subplots()
plt.errorbar(data["field 0°"], data["n_s 0°"], yerr=data["n_s_error 0°"], label=r"$n_s(0°)$", color="red", fmt="o")
plt.errorbar(data["field 90°"], data["n_s 90°"], yerr=data["n_s_error 90°"], label=r"$n_s(90°)$", color="black", fmt="o")

ax.set_title("4.9 GHz Resonator, 45°")
ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$n_s$")
ax.legend()
plt.show()

#%%
data_folder = Path("Data/")

file_path = data_folder / 'n_s_5_7GHz.dat'

data = pd.read_table(file_path, dtype=float,
                     header=0, sep='\t\t',
                     names=["field 0°", "n_s 0°", "n_s_error 0°", "field 90°",
                            "n_s 90°", "n_s_error 90°"],
                     engine='python')

fig, ax = plt.subplots()
plt.errorbar(data["field 0°"], data["n_s 0°"], yerr=data["n_s_error 0°"], label=r"$n_s(0°)$", color="red", fmt="o")
plt.errorbar(data["field 90°"], data["n_s 90°"], yerr=data["n_s_error 90°"], label=r"$n_s(90°)$", color="black", fmt="o")

ax.set_title("5.7 GHz Resonator, 0°")
ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$n_s$")
ax.legend()


plt.show()

#%% Looking cuadratic dependence
fig, ax = plt.subplots()

ax.scatter(data["field 0°"][:]**2, data["n_s 0°"][:],
           label=r"$n_s(0°)$", color="red")
a, b = np.polyfit(data["field 0°"][:8]**2,
                              data["n_s 0°"][:8], deg=1)
f = lambda x: a * x + b
ax.plot(data["field 0°"][:8]**2, f(data["field 0°"][:8]**2), "b--")

ax.scatter(data["field 90°"][:]**2, data["n_s 90°"][:],
           label=r"$n_s(90°)$", color="black")
m, n = np.polyfit(data["field 90°"][:15]**2,
                           data["n_s 90°"][:15], deg=1)
h = lambda x: m * x + n
ax.plot(data["field 90°"][:15]**2, h(data["field 90°"][:15]**2), "--")

ax.set_title("5.7 GHz Resonator, 0°")
ax.set_xlabel(r"$B^2$ [$T^2$]")
ax.set_ylabel(r"$n_s$")
ax.legend()


plt.show()

#%% Interpolation to theory

data_folder = Path("../Density-of-states/Data/")
# data_folder = Path("../anisotropic-superfluid/Data/")

file_to_open = data_folder / "n_By_mu_-349.0_L=2500_h=0.001_B_y_in_(0.0-1.2)_Delta=0.2_lambda_R=1.4049144729009981_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24.npz"
# file_to_open = data_folder / "n_By_mu_-30_L=1000_h=0.01_B_y_in_(0.0-1.2)_Delta=0.2_lambda_R=0.56_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_theta=1.57_points=24.npz"

Data = np.load(file_to_open)

n_B_y = Data["n_B_y"]
B_values = Data["B_values"]
# Lambda = Data["Lambda"]
Lambda_R = Data["Lambda_R"]
# Lambda_D = Data["Lambda_D"]
Delta = Data["Delta"]
theta = Data["theta"]
w_0 = Data["w_0"]
mu = Data["mu"]
L_x = Data["L_x"]
h = Data["h"]
points = Data["points"]
g_xx = Data["g_xx"]
g_yy = Data["g_yy"]
g_xy = Data["g_xy"]
g_yx = Data["g_yx"]

def interpolation_for_theory(x):
    return [
            np.interp(x/1.7, B_values/Delta, n_B_y[:, 0]),        #I have change x to x/2
            np.interp(x, B_values/Delta, n_B_y[:, 1])
            ]



fig, ax = plt.subplots()
ax.plot(B_values/Delta, n_B_y[:, 0], "-sk",  label=r"$n_s(90°)$")
ax.plot(B_values/Delta, n_B_y[:, 1], "-or",  label=r"$n_s(0°)$")
ax.plot(B_values/Delta, interpolation_for_theory(B_values/Delta)[0])
ax.plot(B_values/Delta, interpolation_for_theory(B_values/Delta)[1])

ax.set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{np.round(theta,2)}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"
             +f"; h={h}")
ax.set_xlabel(r"$B$")
ax.set_ylabel(r"$n_s$")
ax.legend()
# saving_folder = Path("/home/gabriel/Dropbox/Figures/Fitting to superfluid density")
# file_name = f'interpolation_to_theory_lambda_R={np.round(Lambda_R,2)}_Delta={Delta}_' + \
#             f'mu={mu}_L={L_x}_h={h}_B_y_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_theta={np.round(theta,2)}.png'

# file_path = saving_folder / file_name
# plt.savefig(file_path, dpi=300)
plt.show()

#%% Fitting to experiment
from scipy.optimize import curve_fit


data_folder = Path("Data/")

file_path = data_folder / 'n_s_5_7GHz.dat'

data = pd.read_table(file_path, dtype=float,
                     header=0, sep='\t\t',
                     names=["field 0°", "n_s 0°", "n_s_error 0°", "field 90°",
                            "n_s 90°", "n_s_error 90°"],
                     engine='python')

# data = data.replace('nan', np.nan)   #replace nan to np.nan
# data = data.dropna()  #remove NaN
# def model_parallel(x, a, b, c, d):
#     return a*(interpolation_for_theory(b*x)[1]) + c + d * x**2

def model_parallel(x, a, c):
    # return a*(interpolation_for_theory(x)[1]) + b + c* x**2
    return a*(interpolation_for_theory(x)[1] - interpolation_for_theory(0)[1]) + data["n_s 0°"][0] + c* x**2

def model_perpendicular(x, a, c):
    # return a*(interpolation_for_theory(x)[0]) + b + c* x**2
    return a*(interpolation_for_theory(x)[0] - interpolation_for_theory(0)[0])  + data["n_s 90°"].dropna()[0] + c* x**2

initial_parameters_parallel = [ 2.10435188e+06, -1.26647832e+03]
popt_parallel, pcov_parallel = curve_fit(model_parallel, data["field 0°"]/data["field 0°"][8], data["n_s 0°"],
                                          p0=initial_parameters_parallel)


initial_parameters_perpendicular = [ 5.39729364e+06, -8.91725182e+02]
popt_perpendicular, pcov_perpendicular = curve_fit(
                                                   model_perpendicular, data["field 90°"].dropna()/data["field 90°"][8], data["n_s 90°"].dropna(),
                                                   p0=initial_parameters_perpendicular
                                                   )

fig, ax = plt.subplots()
ax.errorbar(data["field 0°"]/data["field 0°"][8], data["n_s 0°"], yerr=data["n_s_error 0°"], label=r"$n_s(0°)$", color="red", fmt="o", zorder=1)
ax.errorbar(data["field 90°"].dropna()/data["field 90°"][8], data["n_s 90°"].dropna(), yerr=data["n_s_error 90°"].dropna(), label=r"$n_s(90°)$", color="black", fmt="s", zorder=2)


x_model_parallel  = data["field 0°"]/data["field 0°"][8]
# fig, ax = plt.subplots()
ax.plot(x_model_parallel, model_parallel(x_model_parallel, *popt_parallel), "-b",  label=r"fit of $n_s(0°)$", zorder=3)

x_model_perpendicular  = data["field 90°"].dropna()/data["field 90°"][8]
ax.plot(x_model_perpendicular, model_perpendicular(x_model_perpendicular, *popt_perpendicular), "--g",  label=r"fit of $n_s(90°)$")

ax.set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $\theta=$" + f"{np.round(theta,2)}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"
             +f"; h={h}")

ax.set_xlabel(r"$g\mu_B B/\Delta_0$")
ax.set_ylabel(r"$n_s$")
ax.legend()
plt.show()
#%% Save figure
saving_folder = Path("/home/gabriel/Dropbox/Figures/Fitting to superfluid density")
file_name = f'fit_lambda_R={np.round(Lambda_R,2)}_Delta={Delta}_' \
            f'mu={mu}_L={L_x}_h={h}_B_y_in_({np.min(B_values)}-{np.round(np.max(B_values),3)})_theta={np.round(theta,2)}.png'

file_path = saving_folder / file_name
fig.savefig(file_path, dpi=300)

#%%

g = 1
mu_B = 5.79e-5 *1000 # meV/T
Delta = 0.2 # meV induced gap


fig, ax = plt.subplots()
ax.errorbar(data["field 0°"], data["n_s 0°"], yerr=data["n_s_error 0°"], label=r"$n_s(0°)$", color="red", fmt="o", zorder=1)
ax.errorbar(data["field 90°"].dropna(), data["n_s 90°"].dropna(), yerr=data["n_s_error 90°"].dropna(), label=r"$n_s(90°)$", color="black", fmt="s", zorder=2)

B_parallel = x_model_parallel  * data["field 0°"][8]  # T
B_perpendicular = x_model_perpendicular  * data["field 90°"][8] 
ax.plot(B_parallel, model_parallel(x_model_parallel, *popt_parallel), "-b",  label=r"fit of $n_s(0°)$", zorder=3)
ax.plot(B_perpendicular, model_perpendicular(x_model_perpendicular, *popt_perpendicular), "--g",  label=r"fit of $n_s(90°)$")

ax.set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}meV"
             +r"; $\Delta=$" + f"{Delta}meV"
             +r"; $\theta=$" + f"{np.round(theta,2)}"
             + r"; $\mu$"+f"={mu}meV"+ "\n"
             +r"$w_0$"+f"={w_0}meV"
             +r"; $L_x=$"+f"{L_x}"
             +f"; h={h}"
             + r"; $g_{xx}/g_{yy}=0.6$")

ax.set_xlabel(r"$B(T)$")
ax.set_ylabel(r"$n_s$")
ax.legend()
plt.show()