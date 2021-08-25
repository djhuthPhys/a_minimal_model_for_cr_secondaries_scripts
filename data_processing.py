import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator


def plot_e_spectrum(energy, flux, energy_sig, flux_sig, title, color, mfc):
    """
    Plots the input data on a loglog scale with error bars
    :param energy:
    :param flux:
    :param energy_sig:
    :param flux_sig:
    :param title:
    :param color:
    :param mfc
    :return:
    """

    y_ticks = [10 ** tick for tick in range(-8, 1)]
    fig, ax = plt.subplots()
    ax.errorbar(energy, flux, yerr=flux_sig, xerr=energy_sig, fmt='s', color=color, markerfacecolor=mfc)
    ax.set_title(title, size=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Energy (GeV)', fontsize=15)
    ax.set_ylabel('$e^{-}$ flux [(m$^2$ sr s GeV)$^{-1}$]', fontsize=15)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    ax.yaxis.set_major_locator(LogLocator(numticks=15))  # (1)
    ax.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
    fig.set_size_inches(6, 10)
    fig.tight_layout()
    plt.show()


def plot_spectrum(energy, flux, flux_sig, title, color, mfc, sym='o'):
    """
        Plots the input data on a loglog scale with error bars
        :param energy:
        :param flux:
        :param flux_sig:
        :param title:
        :param color:
        :param mfc:
        :param sym: option for defining the symbol representing the data
        :return:
        """

    # y_ticks = [10 ** tick for tick in range(-8, 1)]
    fig, ax = plt.subplots()
    ax.errorbar(energy, flux, yerr=flux_sig, xerr=None, fmt=sym, color=color, markerfacecolor=mfc)
    ax.set_title(title, size=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rigidity (GV)', fontsize=15)
    ax.set_ylabel(r'B Flux [(m$^2$ sr s GV)$^{-1}$]', fontsize=15)
    # ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    ax.yaxis.set_major_locator(LogLocator(numticks=15))  # (1)
    ax.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
    fig.set_size_inches(6, 10)
    fig.tight_layout()
    plt.show()

    return fig, ax


def plot_p_bar_p(pos=False):
    """
    Plots the anti-proton to proton ratio. Option to add the positron to proton ratio to plot in pos
    :return:
    """
    color = 'green'
    mfc = '#BAF282'
    symbol = 's'

    # data = pd.read_csv('./data/AMS_2_anti_proton_data.csv')
    # rigidity = data['Rigidity Low']
    # ratio = data['Ratio'] * data['Ratio order']
    # ratio_sig = data['Total sig_R'] * data['Ratio order']

    extra_data = pd.read_csv('./data/AMS_2_boron_carbon_plus_oxygen_ratio.csv')
    rigidity = extra_data['Rigidity']
    ratio = extra_data['Ratio']
    ratio_sig = extra_data['Total sig']

    fig, ax = plt.subplots()
    ax.errorbar(rigidity, ratio, yerr=ratio_sig, xerr=None, fmt=symbol, color=color, markerfacecolor=mfc, markersize=5)
    # if pos:
        # ax.errorbar(rigidity_e, ratio_e, yerr=ratio_sig_e, xerr=None, fmt='^', color='red', markerfacecolor='#ff8282',
        #             markersize=5)
        # ax.legend([r'$\bar{p}/p$', r'$e^{+}/p$'])
    ax.set_title(r'AMS 02 B/(C+O) Ratio', size=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rigidity (GV)', fontsize=15)
    ax.set_ylabel('Ratio', fontsize=15)
    ax.set_xlim(left=1)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    ax.yaxis.set_major_locator(LogLocator(numticks=15))  # (1)
    ax.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
    fig.set_size_inches(6, 10)
    fig.tight_layout()
    plt.savefig('./plots/test.pdf')
    plt.show()


def multi_plot(species_dict):
    """
    Plots flux data from different particle species on the same plot
    :param species_dict: dictionary of species and symbol/color options to use in plotting
                         {species name:[symbol, primary color, secondary color, fill style, marker size, legend label]}
    :return:
    """

    fig, ax = plt.subplots()
    for key in species_dict:
        print('Plotting ' + key + ' flux')
        data = pd.read_csv('./data/AMS_2_' + key +'_data.csv')
        symbol = species_dict[key][0]
        border_color = species_dict[key][1]
        face_color = species_dict[key][2]
        fst = species_dict[key][3]
        size = species_dict[key][4]

        if key in ('electron', 'positron'):
            x = data['E']
            y = data['Flux'] * data['Order']
            y_sig = data['Total sig'] * data['Order']
        else:
            x = (data['Rigidity Low'] + data['Rigidity High'])/2
            y = data['Flux'] * data['Order']
            y_sig = data['Total sig'] * data['Order']

        ax.errorbar(x, y, yerr=y_sig, xerr=None, fmt=symbol, color=border_color, markerfacecolor=face_color,
                    fillstyle=fst, markersize=size)
    ax.legend([species_dict[key][5] for key in species_dict])
    ax.set_title('AMS 02 Flux', size=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rigidity (GV)', fontsize=15)
    ax.set_ylabel('Flux [(m$^2$ sr s GV)$^{-1}$]', fontsize=15)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    ax.yaxis.set_major_locator(LogLocator(numticks=15))  # (1)
    ax.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
    fig.set_size_inches(6, 10)
    fig.tight_layout()
    plt.show()


def norm_multi_plot(species_dict, norm_species):
    """
     Plots flux data from different particle species on the same plot normalized at the energy/rigidity specified by
     norm_species
    :param species_dict:
    :param norm_species:
    :return:
    """
    # Get flux of norm_species at 100 GV for normalization
    norm_data = pd.read_csv('./data/AMS_2_' + norm_species +'_data.csv')
    rigidity = norm_data['Rigidity Low']
    flux = norm_data['Flux'] * norm_data['Order']
    norm_idx = rigidity.index[rigidity == 100].item()
    norm_flux = flux[norm_idx]

    fig, ax = plt.subplots()
    for key in species_dict:
        print('Plotting ' + key + ' flux')
        data = pd.read_csv('./data/AMS_2_' + key +'_data.csv')
        symbol = species_dict[key][0]
        border_color = species_dict[key][1]
        face_color = species_dict[key][2]
        fst = species_dict[key][3]
        size = species_dict[key][4]

        x = (data['Rigidity Low'] + data['Rigidity High'])/2
        y = (data['Flux'] * data['Order'])/norm_flux
        y_sig = data['Total sig'] * data['Order']

        ax.errorbar(x, y, yerr=y_sig, xerr=None, fmt=symbol, color=border_color, markerfacecolor=face_color,
                    fillstyle=fst, markersize=size)
    ax.legend([species_dict[key][5] for key in species_dict])
    ax.set_title('AMS 02 Normalized Flux', size=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rigidity (GV)', fontsize=15)
    ax.set_ylabel('Normalized Flux [(m$^2$ sr s GV)$^{-1}$]', fontsize=15)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    ax.yaxis.set_major_locator(LogLocator(numticks=15))  # (1)
    ax.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
    fig.set_size_inches(6, 10)
    fig.tight_layout()
    plt.show()


def calculate_ratios(primary_species, secondary_species):
    """
    Calculates the flux ratio primary/secondary and the corresponding uncertainties. ASSUMES BINNING IS THE SAME
    :param primary_species: string of primary species ex: 'carbon'
    :param secondary_species: string of secondary species
    :return:
    """

    primary_data = pd.read_csv('./data/AMS_2_' + primary_species + '_data.csv')
    secondary_data = pd.read_csv('./data/AMS_2_' + secondary_species + '_data.csv')

    # Calculate ratios
    ratios = pd.DataFrame((primary_data['Flux']* primary_data['Order'].values) \
            /(secondary_data['Flux'].values*secondary_data['Order'].values))

    # Calculate errors
    primary_sig = primary_data[['Syst sig']]\
                  *primary_data[['Order']].values
    secondary_sig = secondary_data[['Syst sig']]\
                    *secondary_data[['Order']].values

    ratio_sig = ratios[['Flux']].values\
                * np.sqrt((primary_sig/primary_data[['Flux']].values)**2
                          + (secondary_sig.values/secondary_data[['Flux']].values)**2)

    # Create and save final dataframe to save
    ratios.columns = ['Ratio']
    print(ratios)
    ratio_df = pd.concat([primary_data[['Rigidity Low','Rigidity High']], ratios, ratio_sig], axis=1)
    ratio_df.to_csv('./data/AMS_2_' + primary_species + '_' + secondary_species + '_ratio.csv')

    return ratio_df


def ratio_plot(species_dict):
    """

    :param species_dict: dictionary of species and symbol/color options to use in plotting
                         {species name:[ratio species, symbol, primary color, secondary color, fill style, marker size,
                          legend label]}
    :return:
    """

    fig, ax = plt.subplots()
    for key in species_dict:

        print('Plotting ' + re.sub(r'[^a-zA-Z]', '', key) + '/' + species_dict[key][0] + ' flux ratio')
        primary_data = pd.read_csv('./data/AMS_2_' + re.sub(r'[^a-zA-Z]', '', key) + '_' +
                                     species_dict[key][0] + '_ratio.csv')
        symbol = species_dict[key][1]
        border_color = species_dict[key][2]
        face_color = species_dict[key][3]
        fst = species_dict[key][4]
        size = species_dict[key][5]

        # Define data to plot
        print(re.sub(r'[^a-zA-Z]', '', key))
        if re.sub(r'[^a-zA-Z]', '', key) == 'positron' and species_dict[key][0] == 'proton':
            x = primary_data['Rigidity']
        else:
            x = (primary_data['Rigidity Low'] + primary_data['Rigidity High']) / 2
        y = primary_data['Ratio']
        y_sig = primary_data['Total sig']

        ax.errorbar(x, y, yerr=y_sig, xerr=None, fmt=symbol, color=border_color, markerfacecolor=face_color,
                    fillstyle=fst, markersize=size)

    ax.legend([species_dict[key][6] for key in species_dict])
    ax.set_title(r'AMS 02 FLux Ratios', size=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Rigidity (GV)', fontsize=15)
    ax.set_ylabel('Flux Ratio', fontsize=15)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
    ax.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=15)
    ax.yaxis.set_major_locator(LogLocator(numticks=15))  # (1)
    ax.yaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
    fig.set_size_inches(6, 10)
    fig.tight_layout()
    plt.show()


def combo_plot(species_type):
    """
    Plots the linear combinations of primary or secondary cosmic ray fluxes
    :param species_type: string containing either 'primary' or 'secondary' and determines which flux is plotted
    :return:
    """
    data1 = None
    data2 = None
    data3 = None
    data4 = None
    data5 = None


    if species_type == 'primary':
        data1 = pd.read_csv('./data/AMS_2_carbon_data.csv')
        data2 = pd.read_csv('./data/AMS_2_oxygen_data.csv')
        color = 'red'
        mfc = '#ff8282'

        rigidity = (data1['Rigidity Low'][1:].values + data1['Rigidity High'][1:].values)/2
        flux = (data1['Flux'][1:].values * data1['Order'][1:].values) + (data2['Flux'].values * data2['Order'].values)
        flux_sig = np.sqrt(np.power(data1['Total sig'][1:].values * data1['Order'][1:].values,2) +
                           np.power(data2['Total sig'].values * data2['Order'].values,2))
        plot_spectrum(rigidity, flux, flux_sig, 'AMS-02 C + O ("M") Flux', color, mfc)

    elif species_type == 'secondary':
        data1 = pd.read_csv('./data/AMS_2_lithium_data.csv')
        data2 = pd.read_csv('./data/AMS_2_beryllium_data.csv')
        data3 = pd.read_csv('./data/AMS_2_boron_data.csv')
        color = 'blue'
        mfc = '#82CFFF'

        rigidity = (data1['Rigidity Low'] + data1['Rigidity High'])/2
        flux = (data1['Flux'].values * data1['Order'].values) + \
               (data2['Flux'].values * data2['Order'].values) + \
               (data3['Flux'].values * data3['Order'].values)
        flux_sig = np.sqrt(np.power(data1['Total sig'].values * data1['Order'].values,2) +
                           np.power(data2['Total sig'].values * data2['Order'].values,2) +
                           np.power(data3['Total sig'].values * data3['Order'].values,2))

        plot_spectrum(rigidity, flux, flux_sig, 'AMS-02 Li + Be + B ("L") Flux', color, mfc)

    elif species_type == 'ratio':
        data1 = pd.read_csv('./data/AMS_2_carbon_data.csv')
        data2 = pd.read_csv('./data/AMS_2_oxygen_data.csv')
        data3 = pd.read_csv('./data/AMS_2_lithium_data.csv')
        data4 = pd.read_csv('./data/AMS_2_beryllium_data.csv')
        data5 = pd.read_csv('./data/AMS_2_boron_data.csv')

        color = 'green'
        mfc = '#BAF282'

        # Consolidate last two energy bins in data1 and data2
        data1['Flux'] = data1['Flux'] * data1['Order']
        data2['Flux'] = data2['Flux'] * data2['Order']
        data3['Flux'] = data3['Flux'] * data3['Order']
        data4['Flux'] = data4['Flux'] * data4['Order']
        data5['Flux'] = data5['Flux'] * data5['Order']

        data1['Total sig'] = data1['Total sig'] * data1['Order']
        data2['Total sig'] = data2['Total sig'] * data2['Order']
        data3['Total sig'] = data3['Total sig'] * data3['Order']
        data4['Total sig'] = data4['Total sig'] * data4['Order']
        data5['Total sig'] = data5['Total sig'] * data5['Order']

        data1_last_flux = data1['Flux'][66] + data1['Flux'][67]

        data1_last_flux_sig = np.sqrt(data1['Total sig'][66] ** 2 + data1['Total sig'][67] ** 2)

        data2_last_flux = data2['Flux'][65] + data2['Flux'][66]

        data2_last_flux_sig = np.sqrt(data2['Total sig'][65] ** 2 + data2['Total sig'][66] ** 2)


        data1['Rigidity High'][66] = data1['Rigidity High'][67]
        data1['Flux'][66] = data1_last_flux
        data1['Total sig'][66] = data1_last_flux_sig

        data2['Rigidity High'][65] = data2['Rigidity High'][66]
        data2['Flux'][65] = data2_last_flux
        data2['Total sig'][65] = data2_last_flux_sig

        data1 = data1[:][0:len(data1) - 1]
        data2 = data2[:][0:len(data2) - 1]

        rigidity = (data3['Rigidity Low'][1:] + data3['Rigidity High'][1:])/2

        L_flux = data3['Flux'][1:].values + data4['Flux'][1:].values + data5['Flux'][1:].values

        L_flux_sig = np.sqrt(np.power(data3['Total sig'][1:].values, 2) +
                             np.power(data4['Total sig'][1:].values, 2) +
                             np.power(data5['Total sig'][1:].values, 2))

        M_flux = data1['Flux'][1:].values + data2['Flux'].values

        M_flux_sig = np.sqrt(np.power(data1['Total sig'][1:].values, 2) + np.power(data2['Total sig'].values, 2))

        ratio = L_flux/M_flux
        ratio_sig = ratio*(L_flux_sig/L_flux + M_flux_sig/M_flux)

        plot_spectrum(rigidity, ratio, ratio_sig, 'AMS-02 L/M Ratio', color, mfc)

    else:
        print('Invalid species type!')

    return data1,data2,data3,data4,data5


def main():

    if plot_all:
        # Load and plot positron spectrum
        data = pd.read_csv('./data/AMS_2_positron_data.csv')
        color = 'green'
        mfc = '#BAF282'

        pos_energy = data['E']
        pos_energy_sig = data['E sig']
        pos_flux = data['Flux'] * data['Order']
        pos_flux_sig = data['Total sig'] * data['Order']

        plot_e_spectrum(pos_energy, pos_flux, pos_energy_sig, pos_flux_sig, 'AMS-02 Positron Flux', color, mfc)

        # Load and plot electron spectrum
        data = pd.read_csv('./data/AMS_2_electron_data.csv')
        color = 'blue'
        mfc = '#82CFFF'

        ele_energy = data['E']
        ele_energy_sig = data['E sig']
        ele_flux = data['Flux'] * data['Order']
        ele_flux_sig = data['Total sig'] * data['Order']

        plot_e_spectrum(ele_energy, ele_flux, ele_energy_sig, ele_flux_sig, 'AMS-02 Electron Flux', color, mfc)

        # Load and plot proton spectrum
        data = pd.read_csv('./data/AMS_2_proton_data.csv')
        color = 'red'
        mfc = '#ff8282'

        pro_rigidity = (data['Rigidity Low'] + data['Rigidity High'])/2
        pro_flux = data['Flux'] * data['Order']
        pro_flux_sig = data['Total sig'] * data['Order']

        plot_spectrum(pro_rigidity, pro_flux, pro_flux_sig, 'AMS-02 Proton Flux', color, mfc)

        # Load and plot antiproton spectrum
        data = pd.read_csv('./data/AMS_2_anti_proton_data.csv')
        color = 'purple'
        mfc = '#be82ff'

        anti_rigidity = (data['Rigidity Low'] + data['Rigidity High'])/2
        anti_flux = data['Flux'] * data['Order']
        anti_flux_sig = data['Total sig'] * data['Order']

        plot_spectrum(anti_rigidity, anti_flux, anti_flux_sig, 'AMS-02 Anti-proton Flux', color, mfc)

        # Load and plot Helium spectrum
        data = pd.read_csv('./data/AMS_2_helium_data.csv')
        color = 'orange'
        mfc = '#ffc182'

        he_rigidity = (data['Rigidity Low'] + data['Rigidity High'])/2
        he_flux = data['Flux'] * data['Order']
        he_flux_sig = data['Total sig'] * data['Order']

        plot_spectrum(he_rigidity, he_flux, he_flux_sig, 'AMS-02 Helium Flux', color, mfc)

        # Load and plot Carbon spectrum
        data = pd.read_csv('./data/AMS_2_carbon_data.csv')
        color = 'green'
        mfc = '#BAF282'

        he_rigidity = (data['Rigidity Low'] + data['Rigidity High'])/2
        he_flux = data['Flux'] * data['Order']
        he_flux_sig = data['Total sig'] * data['Order']

        plot_spectrum(he_rigidity, he_flux, he_flux_sig, 'AMS-02 Carbon Flux', color, mfc)

        # Load and plot Oxygen spectrum
        data = pd.read_csv('./data/AMS_2_oxygen_data.csv')
        color = 'red'
        mfc = '#ff8282'

        he_rigidity = (data['Rigidity Low'] + data['Rigidity High'])/2
        he_flux = data['Flux'] * data['Order']
        he_flux_sig = data['Total sig'] * data['Order']

        plot_spectrum(he_rigidity, he_flux, he_flux_sig, 'AMS-02 Oxygen Flux', color, mfc)

        # Load and plot Lithium spectrum
        data = pd.read_csv('./data/AMS_2_lithium_data.csv')
        color = 'blue'
        mfc = '#82CFFF'

        he_rigidity = (data['Rigidity Low'] + data['Rigidity High'])/2
        he_flux = data['Flux'] * data['Order']
        he_flux_sig = data['Total sig'] * data['Order']

        plot_spectrum(he_rigidity, he_flux, he_flux_sig, 'AMS-02 Lithium Flux', color, mfc)

        # Load and plot Beryllium spectrum
        data = pd.read_csv('./data/AMS_2_beryllium_data.csv')
        color = 'purple'
        mfc = '#be82ff'

        he_rigidity = (data['Rigidity Low'] + data['Rigidity High'])/2
        he_flux = data['Flux'] * data['Order']
        he_flux_sig = data['Total sig'] * data['Order']

        plot_spectrum(he_rigidity, he_flux, he_flux_sig, 'AMS-02 Beryllium Flux', color, mfc)

        # # Load and plot Boron spectrum
        data = pd.read_csv('./data/AMS_2_boron_data.csv')
        color = 'orange'
        mfc = '#ffc182'

        he_rigidity = (data['Rigidity Low'] + data['Rigidity High'])/2
        he_flux = data['Flux'] * data['Order']
        he_flux_sig = data['Total sig'] * data['Order']

        plot_spectrum(he_rigidity, he_flux, he_flux_sig, 'AMS-02 Boron Flux', color, mfc)

    if multiplot:
        # # Plot secondary CRs on same plot (Li, Be, B)
        # secondary_dict = {'lithium': ['s', 'red', '#ff8282', 'full', 5, 'Li'],
        #                   'beryllium': ['o', 'blue', '#82cfff', 'full', 5, 'Be'],
        #                   'boron': ['^', 'green', '#baf282', 'full', 5, 'B']}
        #
        # multi_plot(secondary_dict)
        #
        # # Plot primary Crs on same plot (C, O)
        # primary_dict = {'carbon': ['o', 'red', '#ff8282', 'full', 7, 'C'],
        #                 'oxygen': ['s', 'green', '#baf282', 'full', 5, 'O']}
        #
        # multi_plot(primary_dict)

        # # Plot protons, antiprotons, and positrons
        # particle_dict = {'proton': ['s', 'red', '#ff8282', 'full', 5, r'$p$'],
        #                   'anti_proton': ['o', 'blue', '#82cfff', 'full', 5, r'$\bar{p}$'],
        #                   'positron': ['^', 'green', '#baf282', 'full', 5, r'$e^+$']}
        #
        # multi_plot(particle_dict)

        # Plot
        particle_dict = {'lithium': ['s', 'red', '#ff8282', 'full', 5, 'Li'],
                         'beryllium': ['o', 'blue', '#82cfff', 'full', 5, 'Be'],
                         'boron': ['^', 'green', '#baf282', 'full', 5, 'B'],
                         'carbon': ['+', 'purple', '#be82ff', 'full', 5, 'C'],
                         'oxygen': ['>', 'orange', '#ffc182', 'full', 5, 'O']}

        multi_plot(particle_dict)

        # Plot normalized secondary CRs on same plot (Li, Be, B)
        # secondary_dict = {'lithium': ['s', 'red', '#ff8282', 'full', 5, 'Li'],
        #                   'beryllium': ['o', 'blue', '#82cfff', 'full', 5, 'Be'],
        #                   'boron': ['^', 'green', '#baf282', 'full', 5, 'B']}
        #
        # norm_multi_plot(secondary_dict, 'beryllium')
        #
        # # Plot primary CRs on same plot (C, O)
        # primary_dict = {'carbon': ['o', 'red', '#ff8282', 'full', 7, 'C'],
        #                 'oxygen': ['s', 'green', '#baf282', 'full', 5, 'O']}
        #
        # norm_multi_plot(primary_dict, 'carbon')

    if ratioplot:
        # # Plot B/C, Be/C, and Li/C ratio on same plot
        # species_dict = {'boron': ['carbon', '^', 'green', '#baf282', 'full', 5, 'B/C'],
        #                 'lithium': ['carbon', 's', 'red', '#ff8282', 'full', 5, 'Li/C'],
        #                 'beryllium': ['carbon', 'o', 'blue', '#82cfff', 'full', 5, 'Be/C']}
        #
        # ratio_plot(species_dict)
        #
        # # Plot B/O, Be/O, and Li/O ratio on same plot
        # species_dict = {'boron': ['oxygen', '^', 'green', '#baf282', 'full', 5, 'B/O'],
        #                 'lithium': ['oxygen', 's', 'red', '#ff8282', 'full', 5, 'Li/O'],
        #                 'beryllium': ['oxygen', 'o', 'blue', '#82cfff', 'full', 5, 'Be/O']}
        #
        # ratio_plot(species_dict)

        # Plot e+/p, p-/p, ratio on same plot
        species_dict = {'positron': ['proton', '^', 'green', '#baf282', 'full', 5, r'$e^{+}/p$'],
                        'anti_proton': ['proton', 's', 'red', '#ff8282', 'full', 5, r'\bar{p}/p$']}

        ratio_plot(species_dict)


if __name__ == '__main__':
    plot_all = 0
    multiplot = 1
    ratioplot = 0
    main()
