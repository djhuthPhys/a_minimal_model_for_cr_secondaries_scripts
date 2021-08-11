import os

import scipy.integrate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from matplotlib.ticker import LogLocator


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
    print(file_name)
    data_frame = pd.read_csv('./data/' + file_name)

    # Organize data
    if species in ['electron', 'positron']:
        flux = data_frame['Flux'] * data_frame['Order']
        flux_sig = data_frame['Total sig'] * data_frame['Order']
        x = data_frame['E']
    else:
        flux = data_frame['Flux'] * data_frame['Order']
        flux_sig = data_frame['Total sig'] * data_frame['Order']
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
    for i in range(len(indices)-2):
        # adjusted_energy = x[indices[i]:indices[i+1]+1]/((x[indices[i]] + x[indices[i+1]-1])/2)
        params, cov = curve_fit(f=power_law, xdata=x[indices[i+1]:indices[i+2]+1],
                                ydata=flux[indices[i+1]:indices[i+2]+1], p0=[1,3])
        stdevs = np.sqrt(np.diag(cov))
        print(params)
        print(stdevs)
        param_list.append(params)
        std_list.append(stdevs)

    return flux, flux_sig, x, param_list, std_list, indices


def power_law(e, a, beta):
    return a/np.power(e, beta)


def nlb_integrand(t, args):
    b = 1.5*10**(-3)
    tau = 1
    e, beta = args
    integrand = (1-b*e*t)**(beta-2) * (1/np.exp(t/tau))
    return integrand


def nlb_spectrum(e, a, beta):
    factor = (a/(np.power(e, beta)))
    res = scipy.integrate.quad(nlb_integrand, 0, (1 / (1.5 * 10 ** (-3) * e)), [e, beta])
    result = factor * res[0]
    return result


def scan_nlb_integrand(t, args):
    # b = 1.5*10**(-3)
    tau = 1
    e, beta, b = args
    integrand = (1-b*e*t)**(beta-2) * (1/np.exp(t/tau))
    return integrand


def scan_nlb_spectrum(e, a, beta, b):
    factor = (a/(np.power(e, beta)))
    res = scipy.integrate.quad(scan_nlb_integrand, 0, (1 / (b * e)), [e, beta, b])
    result = factor * res[0]
    return result


def edl_exp_integrand(y, args):
    e, tau_o, delta, b = args
    integrand = 1/(tau_o * (e/(1-b*e*y))**-delta)
    return integrand


def edl_integrand(t, args):
    b = 1.5*10**-3
    e, beta, tau_o, delta = args
    exp_term = scipy.integrate.quad(edl_exp_integrand, 0, (1/(b*e)), [e, tau_o, delta, b])
    integrand = (1 - b*e*t)**(beta-2)*(1/np.exp(exp_term[0]))
    return integrand


def edl_spectrum(e, a, beta, tau_o):
    # tau_o = 1*10**6
    delta = 0.11
    factor = a/e**beta
    res = scipy.integrate.quad(edl_integrand, 0, 1/(1.5*10**-3 * e), [e, beta, tau_o, delta])
    spectrum = factor * res[0]
    return spectrum


def test_nlb_model():
    """
    Tests the simple model with a set of b or tau values and returns a plot of the calculated spectra and spectra values
    :return:
    """
    data_frame = pd.read_csv('./data/AMS_2_electron_data.csv')
    flux = data_frame['Flux'] * data_frame['Order']
    flux_sig = data_frame['Total sig'] * data_frame['Order']
    e_vals = data_frame['E']

    # Parameters
    a = 1
    beta = 2.1
    tau_vals = [14.375,]  # np.linspace(1, 10, 10)
    b = 6 * 10 ** (-3)

    # Perform model integration
    spectra = []
    for tau in tau_vals:
        spectrum = []
        for e in e_vals:
            spectrum_val = nlb_spectrum(e, a, beta)
            spectrum.append(spectrum_val)
        spectra.append(spectrum)

    # Plot results
    color = 'green'
    mfc = '#BAF282'
    sym = 's'
    fig, ax = plt.subplots()
    ax.errorbar(e_vals, flux, yerr=flux_sig, xerr=None, fmt=sym, color=color, markerfacecolor=mfc, zorder=-1)
    for spectrum in spectra:
        ax.plot(e_vals, spectrum, color='red')
    ax.set_title('Simple model spectra', size=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Energy (GeV)', fontsize=15)
    ax.set_ylabel('Flux [(m$^2$ sr s GeV)$^{-1}$]', fontsize=15)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    # ax.legend([r'$\tau$ = %.3f' % tau for tau in tau_vals])

    return e_vals, spectra


def test_nlb_fitting(e, flux):
    """

    :param e:
    :param flux:
    :return:
    """
    vcurve = np.vectorize(nlb_spectrum)
    params, cov = curve_fit(f=vcurve, xdata=e, ydata=flux,
                            p0=[1, 2], method='lm')
    print(params)


def fit_nlb_model(species, cuts):
    """
    Fits data of CR species with the simple model described in simple_curve
    :param species:
    :param cuts:
    :return:
    """
    # Get data of species
    file_list = os.listdir('./data')
    file_name = [string for string in file_list if species + '_data' in string][0]
    data_frame = pd.read_csv('./data/' + file_name)

    # Organize data
    if species in ['electron', 'positron']:
        length = len(data_frame['Flux'])-1  # Neglect last point in data due to large errors
        flux = data_frame['Flux'][0:length] * data_frame['Order'][0:length]
        flux_sig = data_frame['Total sig'][0:length] * data_frame['Order'][0:length]
        x = data_frame['E'][0:length]
    else:
        print('Invalid species for this model')
        return None

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
    vcurve = np.vectorize(nlb_spectrum)
    for i in range(len(indices)-2):
        params, cov = curve_fit(f=vcurve, xdata=x[indices[i+1]:indices[i+2]+1], ydata=flux[indices[i+1]:indices[i+2]+1],
                                p0=[1,3], bounds=(-np.inf, np.inf))
        print(params)
        stdevs = np.sqrt(np.diag(cov))
        print(stdevs)
        param_list.append(params)
        std_list.append(stdevs)

    return flux, flux_sig, x, param_list, std_list, indices


def fit_edl_model(species, cuts):
    """
        Fits data of CR species with the simple model described in simple_curve
        :param species:
        :param cuts:
        :return:
        """
    # Get data of species
    file_list = os.listdir('./data')
    file_name = [string for string in file_list if species + '_data' in string][0]
    data_frame = pd.read_csv('./data/' + file_name)

    # Organize data
    if species in ['electron', 'positron']:
        length = len(data_frame['Flux']) - 1  # Neglect last point in data due to large errors
        flux = data_frame['Flux'][0:length] * data_frame['Order'][0:length]
        flux_sig = data_frame['Total sig'][0:length] * data_frame['Order'][0:length]
        x = data_frame['E'][0:length]
    else:
        print('Invalid species for this model')
        return None

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
    vcurve = np.vectorize(edl_spectrum)
    for i in range(len(indices) - 2):
        params, cov = curve_fit(f=vcurve, xdata=x[indices[i + 1]:indices[i + 2] + 1],
                                ydata=flux[indices[i + 1]:indices[i + 2] + 1],
                                p0=[1, 3, 10**6])
        print(params)
        stdevs = np.sqrt(np.diag(cov))
        print(stdevs)
        param_list.append(params)
        std_list.append(stdevs)

    return flux, flux_sig, x, param_list, std_list, indices


def plot_fit_w_data(species, params, curve, plot_options):
    """
    Plots data and the fit line using the parameters in params and energy or rigidity values in x
    :param species:
    :param params:
    :param curve:
    :param plot_options:
    :return:
    """
    # Load full data set
    file_list = os.listdir('./data')
    file_name = [string for string in file_list if species + '_data' in string][0]
    data_frame = pd.read_csv('./data/' + file_name)
    # b_vals = [1.5*10**-3, 3*10**-3, 5*10**-3, 10*10**-3]

    # Organize data
    if species in ['electron', 'positron']:
        flux = data_frame['Flux'] * data_frame['Order']
        flux_sig = data_frame['Total sig'] * data_frame['Order']
        x = data_frame['E']
    else:
        flux = data_frame['Flux'] * data_frame['Order']
        flux_sig = data_frame['Total sig'] * data_frame['Order']
        x = (data_frame['Rigidity Low'] + data_frame['Rigidity High'])/2

    # Generate fit line data
    x_segments = [x]
    fit_lines = []

    if curve == nlb_spectrum:
        vcurve = np.vectorize(curve)
        # for b in b_vals:
        for i in range(len(params)):
            # segment = x[indices[i+1]:indices[i+2]+1]
            fit_line = vcurve(x, params[i][0], params[i][1])
            fit_lines.append(fit_line)
            x_segments.append(x)
    elif curve == edl_spectrum:
        vcurve = np.vectorize(curve)
        # for b in b_vals:
        for i in range(len(params)):
            # segment = x[indices[i+1]:indices[i+2]+1]
            fit_line = vcurve(x, params[i][0], params[i][1], params[i][2])
            fit_lines.append(fit_line)
            x_segments.append(x)

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
    ax.set_xlabel('Energy (GeV)', fontsize=15)
    ax.set_ylabel('Flux [(m$^2$ sr s GeV)$^{-1}$]', fontsize=15)
    ax.set_xlim(left=1)
    ax.set_ylim(top=max(flux)*10)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    # ax.text(100, .1, r'$\beta=%.2f$' % params[0][1])
    # ax.text(100, .03, r'b = %.4f' % params[0][2])
    ax.yaxis.set_major_locator(LogLocator(numticks=15))  # (1)
    ax.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
    # ax.legend(['AMS-02 data', 'b = 0.0000'] + ['b = %.4f' % b for b in b_vals])
    fig.set_size_inches(6,10)
    fig.tight_layout()
    fig.savefig('./plots/test.pdf')
    plt.show()


def main(species, curve, fit_function, cuts=(10,)):
    """
    Fits the data of a CR species using the cuts in cuts and plots the data and fit
    :param species:
    :param curve:
    :param fit_function:
    :param cuts:
    :return:
    """
    plot_options = ['s', 'green', '#baf282', 'AMS 02 ' + species + ' NLB model fit']

    print('Fitting data')
    flux, flux_sig, energy, param_list, std_list, indices = fit_function(species, cuts)

    print('Plotting data')
    plot_fit_w_data(species, param_list, curve, plot_options)


if __name__ == '__main__':
    main('positron', edl_spectrum, fit_edl_model, cuts=(4,))
