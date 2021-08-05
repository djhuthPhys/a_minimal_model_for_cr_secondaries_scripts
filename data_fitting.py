import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

def power_law_fit(species, cuts):
    """
    Fits flux data of a particular species with a power law based on the cuts designated in the cuts list
    :param species: the name of the species to fit
    :param cuts:
    :return:flux, x, param_list, std_list
    """
    # Get data of species
    file_list = os.listdir('./data')
    file_name = [string for string in file_list if species + '_data' in string][0]
    data_frame = pd.read_excel('./data/' + file_name)

    # Organize data
    if species in ['electron', 'positron']:
        flux = data_frame['Flux'] * data_frame['Order']
        flux_sig = data_frame['Syst sig'] * data_frame['Order']
        x = data_frame['E']
    else:
        flux = data_frame['Flux'] * data_frame['Order']
        flux_sig = data_frame['Syst sig'] * data_frame['Order']
        x = (data_frame['Rigidity Low'] + data_frame['Rigidity High'])/2

    # Select data indices for fitting based on cuts
    indices = [0]
    for cut in cuts:
        diffs = np.abs(x - cut)
        index = diffs[diffs == min(diffs)].index[0]
        indices.append(index)
    indices.append(len(flux))

    # Fit data
    param_list = []
    std_list = []
    for i in range(len(indices)-1):
        adjusted_energy = x[indices[i]:indices[i+1]+1]/x[indices[i]]
        params, cov = curve_fit(f=power_law, xdata=adjusted_energy,
                                ydata=flux[indices[i]:indices[i+1]+1], p0=[0,0], bounds=(-np.inf, np.inf))
        stdevs = np.sqrt(np.diag(cov))
        param_list.append(params)
        std_list.append(stdevs)

    return flux, flux_sig, x, param_list, std_list, indices


def power_law(x, a, b):
    return a*np.power(x, b)


def plot_fit_w_data(flux, flux_sig,  x, params, indices, plot_options):
    """
    Plots data and the fit line using the parameters in params and energy or rigidity values in x
    :param flux:
    :param flux_sig:
    :param x:
    :param params:
    :param indices:
    :param plot_options:
    :return:
    """
    # Generate fit line data
    x_segments = []
    fit_lines = []
    for i in range(len(params)):
        segment = x[indices[i]:indices[i+1]+1]
        fit_line = power_law(segment, params[i][0], params[i][1]) / np.power(x[indices[i]], params[i][1])
        fit_lines.append(fit_line)
        x_segments.append(segment)

    # Plot
    sym = plot_options[0]
    color = plot_options[1]
    mfc = plot_options[2]
    title = plot_options[3]

    fig, ax = plt.subplots()
    ax.errorbar(x, flux, yerr=flux_sig, xerr=None, fmt=sym, color=color, markerfacecolor=mfc)
    for j in range(len(fit_lines)):
        ax.errorbar(x_segments[j], fit_lines[j], yerr=None, xerr=None)
    ax.set_title(title, size=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rigidity (GV)', fontsize=15)
    ax.set_ylabel('Flux [(m$^2$ sr s GV)$^{-1}$]', fontsize=15)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    plt.show()
