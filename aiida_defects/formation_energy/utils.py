# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida import orm
from aiida.engine import calcfunction
import numpy as np
import pymatgen
from pymatgen.core.composition import Composition
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure

def generate_defect_structure(host, site_coord, species):
    '''
    To create defective structure at the site_coord in the host structure. species specify the type of defect to be created.
    '''
    structure = host.get_pymatgen_structure()
    defect_structure = structure.copy()
    for atom, sign in species.items():
        if sign == 1:
            defect_structure.append(atom, site_coord)
        else:
            site_index = find_index_of_site(structure, site_coord)
            if site_index == None:
                print('WARNING! the index of the defect site cannot be found')
            defect_structure.remove_sites([site_index])
#    defect_structure.to(filename='tempo.cif')
#    defect_structure =  Structure.from_file('tempo.cif')
    return orm.StructureData(pymatgen=defect_structure)

def find_index_of_site(structure, site_coord):
    #structure = host.get_pymatgen_structure()
    lattice = structure.lattice
    defect_site = PeriodicSite(Element('Li'), site_coord, lattice) # Li is just a dummy element. Any other element also works
    for i, site in enumerate(structure):
        if defect_site.distance(site) < 5E-4:
            return i

def get_vbm(calc_node):
    #N_electron = calc_node.res.number_of_electrons
    N_electron = calc_node.outputs.output_parameters.get_dict()['number_of_electrons']
    vb_index = int(N_electron/2)-1
    vbm = np.amax(calc_node.outputs.output_band.get_array('bands')[:,vb_index])

    return vbm

def is_intrinsic_defect(species, compound):
    """
    Check if a defect is an intrisic or extrinsic defect
    """
    composition = Composition(compound)
    element_list = [atom.symbol for atom in composition]

    for atom in species.keys():
        if atom not in element_list:
            return False
    return True

def get_dopant(species, compound):
    """
    Get the dopant
    """
    composition = Composition(compound)
    element_list = [atom.symbol for atom in composition]
    for atom in species.keys():
        if atom not in element_list:
            return atom
    return 'intrinsic'

def get_defect_and_charge_from_label(calc_label):
    spl = calc_label.split('[')
    defect = spl[0]
    chg = float(spl[1].split(']')[0])
    return defect, chg

@calcfunction
def get_data_array(array):
    data_array = array.get_array('data')
    v_data = orm.ArrayData()
    v_data.set_array('data', data_array)
    return v_data

@calcfunction
def get_defect_formation_energy(defect_data, E_Fermi, chem_potentials, pot_alignments):
    '''
    Computing the defect formation energy with and without electrostatic and potential alignment corrections
    Note: 'E_corr' in the defect_data corresponds to the total correction, i.e electrostatic and potential alignment
    '''
    defect_data = defect_data.get_dict()
    E_Fermi = E_Fermi.get_array('data')
    chem_potentials = chem_potentials.get_dict()
    pot_alignments = pot_alignments.get_dict()

    E_defect_formation = {'uncorrected':{}, 'electrostatic': {}, 'electrostatic and alignment': {}}
    for defect, properties in defect_data.items():
        E_defect_formation['uncorrected'][defect] = {}
        E_defect_formation['electrostatic'][defect] = {}
        E_defect_formation['electrostatic and alignment'][defect] = {}

        for chg in properties['charges'].keys():
            Ef_raw = properties['charges'][chg]['E']-properties['E_host']+float(chg)*(E_Fermi+properties['vbm'])
            for spc, sign in properties['species'].items():
                Ef_raw -= sign*chem_potentials[spc][0]
            Ef_corrected = Ef_raw + properties['charges'][chg]['E_corr']

            E_defect_formation['uncorrected'][defect][str(chg)] = Ef_raw
            E_defect_formation['electrostatic'][defect][str(chg)] = Ef_corrected + float(chg)*pot_alignments[defect][str(chg)]
            E_defect_formation['electrostatic and alignment'][defect][str(chg)] = Ef_corrected

    return orm.Dict(dict=E_defect_formation)

# @calcfunction
# def get_defect_formation_energy(defect_data, E_Fermi, pot_alignments, chem_potentials, compound):

#     defect_data = defect_data.get_dict()
#     #formation_energy_dict = formation_energy_dict.get_dict()
#     E_Fermi = E_Fermi.get_dict()
#     chem_potentials = chem_potentials.get_dict()
#     pot_alignments = pot_alignments.get_dict()
#     compound = compound.value

#     intrinsic_defects = {}
#     for defect, properties in defect_data.items():
#         if is_intrinsic_defect(properties['species'], compound):
#             intrinsic_defects[defect] = properties

#     defect_Ef = {}
#     for dopant, e_fermi in E_Fermi.items():
#         defect_temp = intrinsic_defects.copy()
#         if dopant != 'intrinsic':
#             for defect, properties in defect_data.items():
#                 if dopant in properties['species'].keys():
#                     defect_temp[defect] = properties

#         defect_Ef[dopant] = defect_formation_energy(
#                 defect_temp,
#                 e_fermi,
#                 chem_potentials[dopant],
#                 pot_alignments
#                 )

#     return orm.Dict(dict=defect_Ef)

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def convert_key(key):
    new_key = key.replace('-', 'q')
    if has_numbers(key):
        new_key = 'A'+new_key
        return new_key.replace('.', '_')
    else:
        return new_key

def revert_key(key):
    new_key = key.replace('q', '-')
    if has_numbers(key):
        return new_key[1:]#.replace('_', '.')
    else:
        return new_key

@calcfunction
def store_dict(**kwargs):
    new_dict = {}
    for k, v in kwargs.items():
        new_k = revert_key(k)
        if isinstance(v, orm.Dict):
            # new_dict[k.replace('q', '-')] = v.get_dict()
            d = {key.replace('_', '.'): item for key, item in v.get_dict().items()}
            new_dict[new_k] = d
        if isinstance(v, orm.Float):
            new_dict[new_k] = v.value
        if isinstance(v, orm.ArrayData):
            new_dict[new_k] = v.get_array(v.get_arraynames()[0]).item() # get the value from 0-d numpy array
    return orm.Dict(dict=new_dict)


@calcfunction
def get_defect_data(dopant, compound, defect_info, vbm, E_host_outputs_params, total_correction, **kwargs):

    dopant = dopant.value
    compound = compound.value
    vbm = vbm.value
    E_host = E_host_outputs_params.get_dict()['energy']
    defect_info = defect_info.get_dict()
    total_correction = total_correction.get_dict()

    defect_data = {}
    for defect, properties in defect_info.items():
        if is_intrinsic_defect(properties['species'], compound) or dopant in properties['species'].keys():
            defect_data[defect] = {'N_site': properties['N_site'], 'species': properties['species'], 'charges': {},
                                    'vbm': vbm, 'E_host': E_host}
            for chg in properties['charges']:
                defect_data[defect]['charges'][str(chg)] = {'E_corr': total_correction[defect][str(chg)],
                                                            'E': kwargs[convert_key(defect)+'_'+convert_key(str(chg))].get_dict()['energy']
                                                            }

    return orm.Dict(dict=defect_data)


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
def get_raw_formation_energy(defect_energy, host_energy, chempot_sign, chemical_potential,
                             charge, fermi_energy, valence_band_maximum):
    """
    Compute the formation energy without correction
    """
    chempot_sign = chempot_sign.get_dict()
    chemical_potential = chemical_potential.get_dict()

    e_f_uncorrected = defect_energy.value - host_energy.value + charge.value*(valence_band_maximum.value + fermi_energy.value)
    for specie, sign in chempot_sign.items():
        e_f_uncorrected -= sign*chemical_potential[specie]
    return orm.Float(e_f_uncorrected)


@calcfunction
def get_corrected_formation_energy(e_f_uncorrected, correction):
    """
    Compute the formation energy with correction
    """
    e_f_corrected = e_f_uncorrected + correction
    return e_f_corrected

@calcfunction
def get_corrected_aligned_formation_energy(e_f_corrected, defect_charge, alignment):
    """
    Compute the formation energy with correction and aligned
    """
    e_f_corrected_aligned = e_f_corrected - defect_charge * alignment
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
