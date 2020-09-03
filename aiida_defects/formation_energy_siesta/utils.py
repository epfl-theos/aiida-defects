# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.engine import calcfunction


#def run_pw_calculation(pw_inputs, structure, charge):
#    """
#    Run a QuantumESPRESSO PW.x calculation by invoking the appropriate workchain.
#    The user is not restricted in how they set up the PW calculation.
#    This function simply acts as a wrapper to run a user-configured generic pw.x builder object
#
#    Parameters
#    ----------
#    pw_builder : AiiDA ProcessBuilder
#        An AiiDA ProcessBuilder object for the desired QuantumEspresso workchain
#    structure: AiiDA StructureData
#        The required structure
#    charge: AiiDA Float
#        The required total system charge. Adding an electron is negative by convention

#    Returns
#    -------
#    future
#        A future representing the submitted calculation
#    """
#    from aiida.engine import submit
#    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
#    from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
#
#    # Add the appropriate system charge and structure
#    pw_inputs['structure'] = structure
#
#    pw_inputs['parameters']['SYSTEM']['tot_charge'] = charge
#
#    future = submit(PwBaseWorkChain, **pw_inputs)
#
#    return future


#def run_siesta_calculation(pw_inputs, structure, charge):
#    """
#
#    """
#
#    from aiida.engine import submit
#    from aiida_siesta.workflows.base import SiestaBaseWorkChain
#    from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

@calcfunction
def get_raw_formation_energy(defect_energy, host_energy, chemical_potential,
                             charge, fermi_energy, valence_band_maximum):
    """
    Compute the formation energy without correction
    """
    e_f_uncorrected = defect_energy - host_energy - chemical_potential + (
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
