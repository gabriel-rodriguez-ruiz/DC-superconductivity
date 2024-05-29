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
data.plot(x=0, ax=ax, marker="o", markersize=2)
# ax.plot(data_array[:,0], data[:,1], "o", markersize=2, label=r"$\Delta f_{\perp}(3.7 GHz)$")
# ax.plot(data_array[:,0], data[:,2], "o", markersize=2, label=r"$\Delta f_{\parallel}(3.7 GHz)$")
# ax.plot(data_array[:,0], data[:,3], "o", markersize=2, label=r"$\Delta f_{\perp}(4.4 GHz)$")
# ax.plot(data_array[:,0], data[:,4], "o", markersize=2, label=r"$\Delta f_{\parallel}(4.4 GHz)$")

ax.set_xlabel("B [T]")
ax.set_ylabel(r"$\Delta f [Hz]$")
