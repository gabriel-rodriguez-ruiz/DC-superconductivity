#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:55:11 2024

@author: gabriel
"""

import scipy

k_B = scipy.constants.Boltzmann # J/K
mu_B = scipy.constants.physical_constants["Bohr magneton"][0]
T_c = 1.65 # K
g = 11.2
Delta = 1.76 * k_B *T_c

B = (2 * Delta)/(g * mu_B)