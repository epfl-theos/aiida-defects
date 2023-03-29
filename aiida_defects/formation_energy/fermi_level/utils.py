# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.engine import calcfunction
import numpy as np
import random
from pymatgen.core.composition import Composition
from aiida.orm import ArrayData, Float
from scipy.optimize import broyden1
from scipy.optimize.nonlin import NoConvergence
from scipy.special import expit


def compute_net_charge(defect_data, chem_potentials, temperature, unitcell, band_gap, dos_x, dos_y, dopant):
    '''
    This is a nested function that return a function (with E_Fermi as variable) to be use in the
    non-linear solver to obtain the self-consistent Fermi level.

    arguments:
    defect_data : dictionary containing information required to compute the formation energy of each defect
    chem_potentials : dictionary containing the chemical potential of all elements constituting the compound. Each value can be a float or 1d numpy array
    #input_chem_shape : the shape of values of 'chem_potentials' dictionnary. This is needed because we want the code to work both for float or numpy array
    #                for ex. when computing the concentration of a particular defect in the stability region. We can of course do that one
    #                value at a time but it is much slower than vectorization using numpy
    dopant : aliovalent dopants specified by its charge and concentration with the format {'X_1': {'c':, 'q':}, 'X_2': {'c':, 'q':}, ...}.
            Used to compute the change in the defect concentrations with 'frozen defect' approach
    uniticell : is the structure used to compute the Dos, NOT the host supercell used to compute the formation energy
    '''

    dE = dos_x[1] - dos_x[0]
    k_B = 8.617333262145E-05
    convert = 1E24

    def defect_formation_energy(E_Fermi):
        '''
        Compute the defect formation energy of all defects given in the input file as a function of the fermi level
        E_Fermi.
        '''
        E_defect_formation = {}
        for defect in defect_data.keys():
            temp = defect_data[defect]
            Ef = {}
            for chg in temp['charges'].keys():
                E_formation = temp['charges'][chg]['E']-temp['E_host']+float(chg)*(E_Fermi+temp['vbm'])+temp['charges'][chg]['E_corr']
                for spc in temp['species'].keys():
                    E_formation -= temp['species'][spc]*np.array(chem_potentials[spc]) # We convert chem_potentials[spc] to np array because it was converted to list in the calcfunction
                Ef[chg] = E_formation
            E_defect_formation[defect] = Ef
        return E_defect_formation

    def electron_concentration(E_Fermi):
        '''
        Compute electron concentration
        '''
#        if input_chem_shape.ndim != 0:
        upper_dos = np.reshape(dos_y[dos_x>=band_gap], (-1,1)) # For broadcasting
        E_upper = np.reshape(dos_x[dos_x>=band_gap], (-1,1)) # For broadcasting
#        else:
#            upper_dos = dos_y[dos_x>=band_gap]
#            E_upper = dos_x[dos_x>=band_gap]
        temp_n = upper_dos*expit(-(E_upper-E_Fermi)/(k_B*temperature)) # Use expit (expit(x)=1/(1+exp(-x))) to directly compute Fermi-Dirac distribution
        return convert*np.sum(temp_n, axis=0)*dE/unitcell.volume

    def hole_concentration(E_Fermi):
        '''
        Compute hole concentration
        '''
#        if input_chem_shape.ndim != 0:
        lower_dos = np.reshape(dos_y[dos_x<=0.0], (-1,1)) #For broadcasting
        E_lower = np.reshape(dos_x[dos_x<=0.0], (-1,1)) #For broadcasting
#        else:
#            lower_dos = dos_y[dos_x<=0.0]
#            E_lower = dos_x[dos_x<=0.0]
        temp_p = lower_dos*expit(-(E_Fermi-E_lower)/(k_B*temperature)) # Use expit to directly compute Fermi-Dirac distribution
        return convert*np.sum(temp_p, axis=0)*dE/unitcell.volume

#    def electron_concentration(E_Fermi):
#        '''
#        Compute electron concentration
#        '''
#        upper_dos = dos_y[dos_x>=band_gap]
#        E_upper = dos_x[dos_x>=band_gap]
#
#        ndim = E_Fermi.ndim
#        E_Fermi = np.expand_dims(E_Fermi, axis=ndim) # To broadcast with E_upper
#        mask_n = ((E_upper-E_Fermi)/(k_B*temperature) < 700.0) # To avoid overflow in the exp
#        for i in range(ndim):
#            upper_dos = np.repeat(np.expand_dims(upper_dos, axis=0), E_Fermi.shape[ndim-i-1], axis=0)
#        upper_dos[~mask_n] = 0
#        temp = E_upper-E_Fermi
#        temp[~mask_n] = 0
#        temp_n = upper_dos/(np.exp(temp/(k_B*temperature))+1.0)
#
#        return convert*np.sum(temp_n, axis=ndim)*dE/unitcell.volume
#
#    def hole_concentration(E_Fermi):
#        '''
#        Compute hole concentration
#        '''
#        lower_dos = dos_y[dos_x<=0.0]
#        E_lower = dos_x[dos_x<=0.0]
#
#        ndim = E_Fermi.ndim
#        E_Fermi = np.expand_dims(E_Fermi, axis=ndim) # To broadcast with E_lower
#        mask_p = ((E_Fermi-E_lower)/(k_B*temperature) < 700.0) # To avoid overflow in the exp
#        for i in range(ndim):
#            lower_dos = np.repeat(np.expand_dims(lower_dos, axis=0), E_Fermi.shape[ndim-i-1], axis=0)
#        lower_dos[~mask_p] = 0
#        temp = E_Fermi-E_lower
#        temp[~mask_p] = 0
#        temp_p = lower_dos/(np.exp(temp/(k_B*temperature))+1.0)
#
#        return convert*np.sum(temp_p, axis=ndim)*dE/unitcell.volume

    def c_defect(N_site, Ef):
        '''
        compute the concentration of defects having formation energy Ef and can exist in N_sites in the unitcell
        '''
        return convert*N_site*np.exp(-1.0*Ef/(k_B*temperature))/unitcell.volume

    def Net_charge(E_Fermi):
        '''
        compute the total charge of the system. The self-consistent Fermi level is the one for which this net (or total) charge is zero.
        '''
        n = electron_concentration(E_Fermi)
        p = hole_concentration(E_Fermi)
        E_defect_formation = defect_formation_energy(E_Fermi)
        # print(n, p)
        # positive_charge = np.zeros(4)
        # negative_charge = np.zeros(4)
        positive_charge = 0.0
        negative_charge = 0.0
        for key in E_defect_formation.keys():
            for chg in E_defect_formation[key]:
                # print(chg)
                if float(chg) > 0:
                    positive_charge += float(chg)*c_defect(defect_data[key]['N_site'], E_defect_formation[key][chg])
                else:
                    negative_charge += float(chg)*c_defect(defect_data[key]['N_site'], E_defect_formation[key][chg])
        if dopant != None:
            for key in dopant.keys():
                if dopant[key]['q'] > 0:
                    positive_charge += dopant[key]['q']*dopant[key]['c']
                else:
                    negative_charge += dopant[key]['q']*dopant[key]['c']
        return np.log(p + positive_charge) - np.log(n + abs(negative_charge))

    return Net_charge

@calcfunction
def solve_for_sc_fermi(defect_data, chem_potentials, temperature, unitcell, band_gap, dos_x, dos_y, dopant, f_tol):
    '''
    solve the non-linear equation with E_fermi as variable to obtain the self-consistent Fermi level. The non-linear solver broyden1 in
    scipy is used.
    '''

    defect_data = defect_data.get_dict()
    chem_potentials =  chem_potentials.get_dict()
    #input_chem_shape = input_chem_shape.get_array('data')
    temperature = temperature.value
    unitcell = unitcell.get_pymatgen_structure()
    band_gap = band_gap.value
    dos_x = dos_x.get_array('data')
    dos_y = dos_y.get_array('data')
    dopant = dopant.get_dict()
    tolerance = f_tol.value

    net_charge = compute_net_charge(defect_data, chem_potentials, temperature, unitcell, band_gap, dos_x, dos_y, dopant)
    #sc_fermi = broyden1(net_charge, input_chem_shape*band_gap/2, f_tol=tolerance)
    input_chem_shape = np.ones_like(random.choice(list(chem_potentials.values())))
    sc_fermi = broyden1(net_charge, input_chem_shape*band_gap/2, f_tol=tolerance)
    v_data = ArrayData()
    v_data.set_array('data', sc_fermi)
    return v_data
