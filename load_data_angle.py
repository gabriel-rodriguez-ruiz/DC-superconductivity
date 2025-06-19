# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 20:20:48 2025

@author: Gabriel
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
#%% Experimental data

data_folder = Path("Data/")

file_path = data_folder / 'n_s_50_mT_rotation.dat'

data = pd.read_table(file_path, dtype=float,
                     header=0, sep='\t\t',
                     names=["angle [°]", "n_s 0°", "n_s_error 0°", "n_s 45°", "n_s_error 45°",
                            "n_s 90°", "n_s_error 90°"],
                     engine='python')

# fig, ax = plt.subplots()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

angle = data["angle [°]"]*2*np.pi/360

plt.errorbar(angle, data["n_s 0°"], yerr=data["n_s_error 0°"], label=r"$n_s(0°)$", color="red", fmt="o")
# plt.errorbar(angle, data["n_s 90°"], yerr=data["n_s_error 90°"], label=r"$n_s(90°)$", color="black", fmt="o")
# plt.errorbar(angle, data["n_s 45°"], yerr=data["n_s_error 45°"], label=r"$n_s(45°)$", color="blue", fmt="o")

ax.set_ylim([2.12e7, 2.14e7])  # n_s 0°
# ax.set_ylim([2.72e7, 2.73e7])    # n_s 90°
# ax.set_ylim([2.97e7, 3e7])    # n_s 45°
# ax.set_ylim([2.1e7, 2.73e7])    # n_s 90° and 0°

ax.set_xlabel(r"$B$ [$T$]")
ax.set_ylabel(r"$n_s$")
ax.legend()
plt.show()


#%% Interpolation to theory

data_folder = Path("../anisotropic-superfluid/Data/")

file_to_open = data_folder / "n_theta_mu_-349.0_L=2500_h=0.001_theta_in_(0.0-1.571)B=0.29_Delta=0.2_lambda_R=1.4049144729009981_lambda_D=0_g_xx=1_g_xy=0_g_yy=1_g_yx=0_points=16.npz"

Data = np.load(file_to_open)

n_theta = Data["n_theta"]
n_theta_0_90 = np.append(
        np.append(
            np.append(
                  n_theta, np.flip(n_theta, axis=0), axis=0), 
                    n_theta, axis=0),
                        np.flip(n_theta, axis=0), axis=0)

# 45°
n_theta_45 = np.append(
        np.append(
            np.append(
                  n_theta, np.flip(-n_theta, axis=0), axis=0), 
                    n_theta, axis=0),
                        np.flip(-n_theta, axis=0), axis=0)

        
theta_values = Data["theta_values"]
theta_values = np.append(np.append(np.append(theta_values, np.pi/2 + theta_values), np.pi + theta_values), 3/2*np.pi + theta_values)

Lambda_R = Data["Lambda_R"]
Lambda_D = Data["Lambda_D"]
Delta = float(Data["Delta"])
w_0 = Data["w_0"]
mu = Data["mu"]
L_x = Data["L_x"]
B = Data["B"]
g_xx = Data["g_xx"]
g_yy = Data["g_yy"]
g_xy = Data["g_xy"]
g_yx = Data["g_yx"]
h = Data["h"]

def interpolation_for_theory(x):
    return [
            np.interp(x, theta_values, n_theta_0_90[:, 0]),        # 0°
            np.interp(x, theta_values, n_theta_0_90[:, 1]),         # 90°
            np.interp(x, theta_values, n_theta_45[:, 2])            # 45°
            ]

# fig, ax = plt.subplots()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

ax.plot(theta_values, n_theta_0_90[:, 0], "-or",  label=r"$n_s(0°)$")
# ax.plot(theta_values, n_theta_0_90[:, 1], "sk",  label=r"$n_s(90°)$")
# ax.plot(theta_values, n_theta_45[:, 2], "-*g",  label=r"$n_s(45°)$")

ax.plot(theta_values, interpolation_for_theory(theta_values)[0])    # 0°
# ax.plot(theta_values, interpolation_for_theory(theta_values)[1])    # 90°
# ax.plot(theta_values, interpolation_for_theory(theta_values)[2])    # 45°


ax.set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; B=" + f"{np.round(B, 2)}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"
             +f"; h={h}")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$n_s$")
ax.legend()
plt.show()

#%% Fitting to experiment
from scipy.optimize import curve_fit

data_folder = Path("Data/")

file_path = data_folder / 'n_s_50_mT_rotation.dat'

data = pd.read_table(file_path, dtype=float,
                     header=0, sep='\t\t',
                     names=["angle [°]", "n_s 0°", "n_s_error 0°", "n_s 45°", "n_s_error 45°",
                            "n_s 90°", "n_s_error 90°"],
                     engine='python')

def model_parallel(x, a, b):
    return a * interpolation_for_theory(x)[0] + b

def model_perpendicular(x, a, b):
    return a * interpolation_for_theory(x)[1] + b

def model_45(x, a, b):
    return a * interpolation_for_theory(x)[2] + b


initial_parameters_parallel = [ 1528777.20922739, 21299892.96422816]
popt_parallel, pcov_parallel = curve_fit(model_parallel, angle, data["n_s 0°"],
                                          p0=initial_parameters_parallel)


initial_parameters_perpendicular = [ 5.39729364e+06, 2.5e7]
popt_perpendicular, pcov_perpendicular = curve_fit(
                                                   model_perpendicular, angle, data["n_s 90°"],
                                                   p0=initial_parameters_perpendicular
                                                   )

initial_parameters_45 = [ 5.39729364e+06, 2.5e7]
popt_45, pcov_45 = curve_fit(model_45, angle, data["n_s 45°"],
                                                   p0=initial_parameters_45
                                                   )

standard_deviation_parallel = np.sqrt(np.diag(pcov_parallel))
standard_deviation_perpendicular = np.sqrt(np.diag(pcov_perpendicular))
standard_deviation_45 = np.sqrt(np.diag(pcov_45))

fig, ax = plt.subplots()
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})


# ax.errorbar(angle, data["n_s 0°"], yerr=data["n_s_error 0°"], label=r"$n_s(0°)$", color="red", fmt="o", zorder=1)
# ax.errorbar(angle, data["n_s 90°"], yerr=data["n_s_error 90°"], label=r"$n_s(90°)$", color="black", fmt="s", zorder=2)
ax.errorbar(angle, data["n_s 45°"], yerr=data["n_s_error 45°"], label=r"$n_s(45°)$", color="blue", fmt="*", zorder=2)


x_model_parallel  = theta_values
# ax.plot(x_model_parallel, model_parallel(x_model_parallel, *popt_parallel), "-b",  label=r"fit of $n_s(0°)$", zorder=3)

x_model_perpendicular  = theta_values
# ax.plot(x_model_perpendicular, model_perpendicular(x_model_perpendicular, *popt_perpendicular), "-g",  label=r"fit of $n_s(90°)$")

x_model_45  = theta_values
ax.plot(x_model_45, model_45(x_model_45, *popt_45), "-r",  label=r"fit of $n_s(45°)$")


ax.set_title(r"$\lambda_R=$" + f"{np.round(Lambda_R,2)}"
             +r"; $\Delta=$" + f"{Delta}"
             +r"; $B=$" + f"{np.round(B,2)}"
             + r"; $\mu$"+f"={mu}"
             +r"; $w_0$"+f"={w_0}"
             +r"; $L_x=$"+f"{L_x}"
             +f"; h={h}"
             +r"; $g_{xx}=$" + f"{g_xx}"
             +r"; $g_{yy}=$" + f"{g_yy}")

# ax.set_ylim([2.12e7, 2.14e7])  # n_s 0°
# ax.set_ylim([2.72e7, 2.73e7])    # n_s 90°
ax.set_ylim([2.97e7, 3e7])    # n_s 45°
# ax.set_ylim([2.1e7, 2.73e7])    # n_s 90° and 0°

# def geometric_factor(x):
#     g_xx = 1
#     g_yy = 1
#     a = min(data["n_s 0°"])
#     b = max(data["n_s 0°"])
#     g = np.sqrt((g_xx*np.cos(x))**2 + (g_yy*np.sin(x))**2)
#     return np.array([a*(g_yy*np.sin(x)/g)**2 + b*(g_xx*np.cos(x)/g)**2,
#                      a*(g_xx*np.sin(x)/g)**2 + b*(g_yy*np.cos(x)/g)**2,
#                      None
#                      ])
# ax.plot(theta_values, [geometric_factor(theta)[1] for theta in theta_values])


ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$n_s$")
plt.tight_layout()
ax.legend()
plt.show()


#%% Save txt

data_folder = Path("Files/")

name = "n_s_50_mT_rotation.txt"
file_to_open = data_folder / name

# data["fit n_s 0°"] = model_parallel(x_model_parallel, *popt_parallel)
# data["fit n_s 90°"] = model_perpendicular(x_model_perpendicular, *popt_perpendicular)
# data["fit n_s 45°"] = model_45(x_model_45, *popt_45)
data["fit n_s 0°"] = model_parallel(angle, *popt_parallel)
data["fit n_s 90°"] = model_perpendicular(angle, *popt_perpendicular)
data["fit n_s 45°"] = model_45(angle, *popt_45)

data.to_csv(file_to_open, sep='\t', header=True, index=False)

#%%
data_folder = Path("Files/")

name = "n_s_50_mT_rotation.txt"
file_to_open = data_folder / name

Data = pd.read_table(file_to_open, dtype=float,
                     header=0, sep='\t',
                     names=['angle [°]', 'n_s 0°', 'n_s_error 0°', 'n_s 45°', 'n_s_error 45°',
                            'n_s 90°', 'n_s_error 90°', 'fit n_s 0°', 'fit n_s 90°', 'fit n_s 45°'],
                     engine='python')

fig, ax = plt.subplots()

ax.plot(Data['angle [°]'], Data["n_s 0°"])
ax.plot(Data['angle [°]'], Data["n_s 90°"])
ax.plot(Data['angle [°]'], Data["n_s 45°"])
ax.plot(Data['angle [°]'], Data["fit n_s 0°"])
ax.plot(Data['angle [°]'], Data["fit n_s 90°"])
ax.plot(Data['angle [°]'], Data["fit n_s 45°"])

