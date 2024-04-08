#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:25:14 2024

@author: gabriel
"""

import numpy as np
from pauli_matrices import tau_0, sigma_0, tau_z, sigma_x, sigma_y, tau_y, tau_x
import scipy
import matplotlib.pyplot as plt

def get_Hamiltonian(k_x, k_y, w_0, mu, Delta, B_x, B_y, Lambda):
    """ Periodic Hamiltonian in x and y.
    """
    return (
            -2*w_0*((np.cos(k_x) + np.cos(k_y))
                   * np.kron(tau_z, sigma_0)
                  ) - mu * np.kron(tau_z, sigma_0)
            + 2*Lambda*(np.sin(k_x) * np.kron(tau_z, sigma_y)
                        - np.sin(k_y) * np.kron(tau_z, sigma_x)
                        )
            - B_x*np.kron(tau_0, sigma_x) - B_y*np.kron(tau_0, sigma_y)
            + Delta*np.kron(tau_x, sigma_0)
            ) * 1/2