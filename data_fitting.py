import os

import scipy.integrate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lmfit import Model
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


def quadratic(e, a, b, c):
    return a + b*e + c*e**2


def cubic(e, a, b, c, d):
    return a + b*e + c*e**2 + d*e**3


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
    e, beta, b, tau = args
    integrand = (1-b*e*t)**(beta-2) * (1/np.exp(t/tau))
    return integrand


def scan_nlb_spectrum(e, a, beta, b, tau):
    factor = (a/(np.power(e, beta)))
    res = scipy.integrate.quad(scan_nlb_integrand, 0, (1 / (b * e)), [e, beta, b, tau])
    result = factor * res[0]
    return result


def edl_exp_integrand(y, args):
    e, tau_o, delta, b = args
    integrand = 1/(tau_o * (e/(1-b*e*y))**-delta)
    return integrand


def edl_integrand(t, args):
    b = 1.6*10**-3
    e, beta, tau_o, delta = args
    exp_term = e**(-1+delta)*(1-b*e*t)**(-delta)*(-1+b*e*t+(1-b*e*t)**delta)/((delta-1)*b*tau_o)
    integrand = (1 - b*e*t)**(beta-2)*np.exp(exp_term)
    return integrand


def edl_spectrum(e, a, beta, delta):
    tau_o = 10**6
    # delta = 0.11
    factor = a/e**beta
    res = scipy.integrate.quad(edl_integrand, 0, 1/(1.6*10**-3 * e), [e, beta, tau_o, delta])
    spectrum = factor * res[0]
    return spectrum


def test_nlb_model():
    """
    Tests the simple model with a set of b or tau values and returns a plot of the calculated spectra and spectra values
    :return:
    """
    data_frame = pd.read_csv('./data/AMS_2_positron_data.csv')
    flux = data_frame['Flux'] * data_frame['Order']
    flux_sig = data_frame['Total sig'] * data_frame['Order']
    e_vals = data_frame['E']

    # Parameters
    a = 8.49
    beta = 2.81
    tau = 1  # np.linspace(1, 10, 10)
    b_vals = [1*10**-3, 1.6*10**-3, 3*10**-3, 5*10**-3, 10*10**-3]
    power_law_spectrum = power_law(e_vals, a, beta)

    # Perform model integration
    spectra = []
    for b in b_vals:
        spectrum = []
        for e in e_vals:
            spectrum_val = scan_nlb_spectrum(e, a, beta, b, tau)/power_law(e, a, beta)
            spectrum.append(spectrum_val)
        spectra.append(spectrum)

    # Plot results
    color = 'green'
    mfc = '#BAF282'
    sym = 's'
    fig, ax = plt.subplots()
    ax.errorbar(e_vals, flux/power_law_spectrum, yerr=flux_sig/power_law_spectrum, xerr=None, fmt=sym, color=color,
                markerfacecolor=mfc, zorder=-1)
    for spectrum in spectra:
        ax.plot(e_vals, spectrum)
    ax.set_title('NLB model to power law ratio', size=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Energy (GeV)', fontsize=15)
    ax.set_ylabel('Ratio', fontsize=15)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    ax.legend([r'b = %.4f' % _ for _ in b_vals])

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
                                p0=[1, 3, 0.5], bounds=[[-np.inf, 2, 0], [np.inf, 3, 1]])
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
    # b_vals = [1.6*10**-3, 3*10**-3, 5*10**-3, 10*10**-3]

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
            fit_line = vcurve(x, params[i][0], params[i][1], params[i][2]) #, params[i][3])
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


def power_law_w_poly_fit(species, cut=(10,)):
    """
    Fits spectrum with a polynomial in log log space at low energies and a power law at higher energies.
    Energy defined by cut.
    :param species:
    :param cut:
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
    diffs = np.abs(x - cut)
    index = diffs[diffs == min(diffs)].index[0]
    indices.append(index)
    indices.append(len(flux))

    # Fit data with power law
    vcurve = np.vectorize(power_law)
    params, cov = curve_fit(f=vcurve, xdata=x[indices[1]:indices[2]+1], ydata=flux[indices[1]:indices[2]+1],
                            p0=[1,3], bounds=(-np.inf, np.inf))
    stdevs = np.sqrt(np.diag(cov))
    q_o = params[0]
    beta = params[1]

    # Use lmfit to constrain quadratic fit to at lower energies
    log_e = np.log(x[indices[0]:indices[1]+1])
    e_cut = x[indices[1]]
    log_flux = np.log(flux[indices[0]:indices[1]+1])

    model = Model(quadratic)
    quad_params = model.make_params(a=0, b=0, c=0)
    quad_params.add('q_o', q_o, vary=False)
    quad_params.add('beta', beta, vary=False)
    quad_params.add('e_cut', e_cut, vary=False)
    quad_params.add('log_e_cut', np.log(e_cut), vary=False)
    # quad_params['a'].expr = '-(q_o/e_cut**beta) - (-b*log_e_cut + c*log_e_cut**2)'
    quad_params['b'].expr = 'q_o*beta/(e_cut**beta) + 2*c*log_e_cut'

    result = model.fit(log_flux, quad_params, e=log_e, method='leastsq', max_nfev=100)
    print(result.fit_report())
    quad_fit_line = quadratic(log_e, result.values['a'], result.values['b'], result.values['c'])
    quad_stdevs = np.sqrt(np.diag(result.covar))
    print(quad_stdevs)
    print(result.values['b'])

    # Generate fit lines
    power_law_line = power_law(x[indices[1]:indices[2]+1], params[0], params[1])

    # Plot results
    sym = 's'
    color = 'green'
    mfc = '#BAF282'
    title = 'AMS-2 Positron with power law and quadratic fit'

    fig, ax = plt.subplots()
    ax.errorbar(x, flux, yerr=flux_sig, xerr=None, fmt=sym, color=color, markerfacecolor=mfc)
    ax.errorbar(x[indices[1]:indices[2]+1], power_law_line, yerr=None, xerr=None)
    ax.errorbar(x[indices[0]:indices[1]+1], np.exp(quad_fit_line), yerr=None, xerr=None)
    ax.set_title(title, size=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Energy (GeV)', fontsize=15)
    ax.set_ylabel('Flux [(m$^2$ sr s GeV)$^{-1}$]', fontsize=15)
    ax.set_xlim(left=1)
    ax.set_ylim(top=max(flux) * 10)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    ax.text(50, 1, r'$\beta=%.3f$' % beta + r'$\pm %.3f$' % stdevs[1])
    ax.text(50, .3, r'$q_{o} = %.3f$' % q_o + r'$\pm %.3f$' % stdevs[0])
    ax.text(50, 0.01, r'a = %.3f' % result.values['a'] + r'$\pm %.3f$' % quad_stdevs[0])
    ax.text(50, 0.003, r'b = %.3f' % result.values['b'] + r'$\pm 0.016$')
    ax.text(50, 0.001, r'c = %.3f' % result.values['c'] + r'$\pm %.3f$' % quad_stdevs[1])
    ax.yaxis.set_major_locator(LogLocator(numticks=15))  # (1)
    ax.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
    # ax.legend(['AMS-02 data', 'b = 0.0000'] + ['b = %.4f' % b for b in b_vals])
    fig.set_size_inches(6, 10)
    fig.tight_layout()
    plt.savefig('./plots/test.pdf')
    plt.show()

    plt.figure()
    plt.plot(np.log(x), np.log(flux), 's')
    plt.plot(np.log(x[indices[0]:indices[1]+1]), result.best_fit)
    plt.show()

    return None


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

