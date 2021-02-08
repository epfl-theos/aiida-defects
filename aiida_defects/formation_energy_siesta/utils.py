# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.engine import calcfunction
import numpy as np

def get_vbm(calc_node):
    N_electron = calc_node.res.number_of_electrons
    vb_index = int(N_electron/2)-1
    vbm = np.amax(calc_node.outputs.output_band.get_array('bands')[:,vb_index])

    return vbm


#@calcfunction
def get_vbm_siesta(calc_node):
    """
    Calculating valence band maximum from siesta calculations"
    """
    import sisl
    
    EIG = sisl.get_sile(calc_node.outputs.remote_folder.get_remote_path()+"/aiida.EIG") 
    
    N_electron = int(EIG.file.read_text().split()[3]) 
    vb_index = int(N_electron/2)-1

    eig_gamma=EIG.read_data()[0][0]
    
    vbm = np.amax(eig_gamma[vb_index])

    return vbm


def get_vbm_siesta_manual(remote_node,host_label):
    """
    Calculating valence band maximum from siesta calculations"
    """
    import sisl

    EIG = sisl.get_sile(remote_node.get_remote_path()+"/"+host_label+".EIG")

    N_electron = int(EIG.file.read_text().split()[3])
    vb_index = int(N_electron/2)-1

    eig_gamma=EIG.read_data()[0][0]

    vbm = np.amax(eig_gamma[vb_index])

    return vbm

def get_vbm_siesta_manual_bands(remote_node,host_label,NE):
    """
    Calculating valence band maximum from siesta calculations"
    """
    import sisl

    BANDS = sisl.get_sile(remote_node.get_remote_path()+"/"+host_label+".bands")

    #N_electron = int(BANDS.file.read_text().split()[3])
    N_electron = NE
    vb_index = int(N_electron/2)-1

    eig_gamma=BANDS.read_data()

    vbm = np.amax(eig_gamma[2][0][0][vb_index])
    #print(vbm)
    return vbm



def run_pw_calculation(pw_inputs, structure, charge):
    """
    Run a QuantumESPRESSO PW.x calculation by invoking the appropriate workchain.
    The user is not restricted in how they set up the PW calculation.
    This function simply acts as a wrapper to run a user-configured generic pw.x builder object

    Parameters
    ----------
    pw_builder : AiiDA ProcessBuilder
        An AiiDA ProcessBuilder object for the desired QuantumEspresso workchain
    structure: AiiDA StructureData
        The required structure
    charge: AiiDA Float
        The required total system charge. Adding an electron is negative by convention

    Returns
    -------
    future
        A future representing the submitted calculation
    """
    from aiida.engine import submit
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
    from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

    # Add the appropriate system charge and structure
    pw_inputs['structure'] = structure

    pw_inputs['parameters']['SYSTEM']['tot_charge'] = charge

    future = submit(PwBaseWorkChain, **pw_inputs)

    return future


@calcfunction
def get_raw_formation_energy(defect_energy, host_energy, add_or_remove, chemical_potential,
                             charge, fermi_energy, valence_band_maximum):
    """
    Compute the formation energy without correction
    """
    # adding none
    sign_of_mu = {'add': +1.0, 'remove': -1.0, 'none' : 0.0}
    e_f_uncorrected = defect_energy - host_energy - sign_of_mu[add_or_remove.value]*chemical_potential + (
        charge * (valence_band_maximum + fermi_energy))
    return e_f_uncorrected


@calcfunction
def get_corrected_formation_energy(e_f_uncorrected, correction):
    """
    Compute the formation energy with correction
    """
    e_f_corrected = e_f_uncorrected + correction
    return e_f_corrected


@calcfunction
def get_corrected_aligned_formation_energy(e_f_corrected, alignment):
    """
    Compute the formation energy with correction and aligned
    """
    e_f_corrected_aligned = e_f_corrected + alignment
    return e_f_corrected_aligned



#@calcfunction
def output_energy_manual(Remote_node):
    """
    Returns Energy from output.out file
    """
    import sisl
    out = sisl.io.outSileSiesta(Remote_node.get_remote_path()+"/output.out")
    f=open(out.file)
    a=f.readlines()
    for line in range(len(a)):
        if 'siesta:         Total =' in a[line]:
            energy = a[line].split()[3]
            #print 
    return energy


def output_total_electrons_manual(Remote_node):
    """
    Return Number of Electrons in System from output.out file 
    """
    import sisl
    out = sisl.io.outSileSiesta(Remote_node.get_remote_path()+"/output.out")
    f=open(out.file)
    a=f.readlines()
    for line in range(len(a)):
        if 'Total number of electrons:' in a[line]:
            number_of_electrons = int(float(a[line].split()[4]))
            #print  (a[line].split())
    return number_of_electrons


# def run_pw_calculation(pw_inputs, charge, run_type, additional_inputs=None):
#     """
#     Run a QuantumESPRESSO PW.x calculation by invoking the PW workchain

#     Parameters
#     ----------
#     pw_inputs : AiiDA Dict
#         A  : AiiDA Float
#         The required total system charge. Adding an electron is negative by convention
#     run_type: AiiDA String
#         The desired type of calculation. Allowed values: 'scf', 'relax', 'vc-relax'

#     Returns
#     -------
#     pw_object?
#         A future representing the submitted calculation
#     """

#     required_keys = [
#         'code', 'pseudos', 'parameters', 'settings', 'metadata',
#         'structure'
#     ]

#     # Validate builder dictionary
#     for key in required_keys:
#         if key not in pw_inputs:
#             raise KeyError(
#                 "Required key, '{}' not found in input dictionary".format(key))

#     # Validate 'run_type'
#     if run_type not in ['scf', 'relax', 'vc-relax']:
#         raise ValueError("Run type, '{}', not recognised".format(run_type))

#     builder['parameters']['SYSTEM']['tot_charge'] = charge

#     if run_type == 'relax' or run_type == 'vc-relax':
#         pw_inputs['relaxation_scheme'] = run_type
#         running = submit(PwRelaxWorkChain, **inputs)
#         self.report(
#             'Launching PwRelaxWorkChain for structure, {}, with charge {} (PK={})'
#             .format(pw_inputs.structure, charge, running.pid))
#         return running
#     else:
#         future = submit(PwBaseWorkChain, **inputs)
#         self.report(
#             'Launching PwBaseWorkChain for structure, {}, with charge {} (PK={})'
#             .format(pw_inputs.structure, charge, future.pid))s
#         return future
