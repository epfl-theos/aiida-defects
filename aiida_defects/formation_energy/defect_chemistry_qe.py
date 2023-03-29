# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np

from aiida import orm
from aiida.engine import WorkChain, calcfunction, ToContext, if_, while_, submit
from aiida.plugins import WorkflowFactory
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_quantumespresso.common.types import RelaxType
from aiida.plugins import CalculationFactory, DataFactory
from aiida.orm.nodes.data.upf import get_pseudos_from_structure

from aiida_defects.formation_energy.defect_chemistry_base import DefectChemistryWorkchainBase
from aiida_defects.formation_energy.utils import run_pw_calculation
#from .utils import get_vbm, get_raw_formation_energy, get_corrected_formation_energy, get_corrected_aligned_formation_energy
from .utils import *
import copy
import pymatgen

PwCalculation = CalculationFactory('quantumespresso.pw')
PpCalculation = CalculationFactory('quantumespresso.pp')
PhCalculation = CalculationFactory('quantumespresso.ph')
DosCalculation = CalculationFactory('quantumespresso.dos')

class DefectChemistryWorkchainQE(DefectChemistryWorkchainBase):
    """
    Compute the formation energy for a given defect using QuantumESPRESSO
    """
    @classmethod
    def define(cls, spec):
        super(DefectChemistryWorkchainQE, cls).define(spec)

        # DFT and DFPT calculations with QuantumESPRESSO are handled with different codes, so here
        # we keep track of things with two separate namespaces. An additional code, and an additional
        # namespace, is used for postprocessing
        spec.input_namespace('qe.dft.supercell',
            help="Inputs for DFT calculations on supercells")
        spec.input_namespace('qe.dft.unitcell',
            help="Inputs for a DFT calculation on an alternative host cell for use DOS and/or DFPT")
        spec.input_namespace('qe.dos',
            help="Inputs for DOS calculation which is needed for the Fermi level workchain")
        spec.input_namespace('qe.dfpt', required=False,
            help="Inputs for DFPT calculation for calculating the relative permittivity of the host material")
        spec.input_namespace('qe.pp',
            help="Inputs for postprocessing calculations")

 #       spec.input('nbands', valid_type=orm.Int,
 #           help="The number of bands used in pw calculation for the unitcell. Need to specify it because we want it to be larger than the default value so that we can get the band gap which is need for the FermiLevelWorkchain.")
        spec.input('k_points_distance', valid_type=orm.Float, required=False, default=lambda: orm.Float(0.2),
            help='distance (in 1/Angstrom) between adjacent kpoints')

        # DFT inputs (PW.x)
        spec.input("qe.dft.supercell.code", valid_type=orm.Code,
            help="The pw.x code to use for the calculations")
        spec.input("qe.dft.supercell.relaxation_scheme", valid_type=orm.Str, required=False,
            default=lambda: orm.Str('relax'),
            help="Option to relax the cell. Possible options are : ['fixed', 'relax', 'vc-relax']")
        #spec.input("qe.dft.supercell.kpoints", valid_type=orm.KpointsData,
        #    help="The k-point grid to use for the calculations")
        spec.input("qe.dft.supercell.parameters", valid_type=orm.Dict,
            help="Parameters for the PWSCF calcuations. Some will be set automatically")
        spec.input("qe.dft.supercell.scheduler_options", valid_type=orm.Dict,
            help="Scheduler options for the PW.x calculations")
        spec.input("qe.dft.supercell.settings", valid_type=orm.Dict,
            help="Settings for the PW.x calculations")
#        spec.input_namespace("qe.dft.supercell.pseudopotentials", valid_type=orm.UpfData, dynamic=True,
#            help="The pseudopotential family for use with the code, if required")
        spec.input("qe.dft.supercell.pseudopotential_family", valid_type=orm.Str,
            help="The pseudopotential family for use with the code")

        # DFT inputs (PW.x) for the unitcell calculation for the dielectric constant
        spec.input("qe.dft.unitcell.code", valid_type=orm.Code,
            help="The pw.x code to use for the calculations")
        spec.input("qe.dft.unitcell.relaxation_scheme", valid_type=orm.Str, required=False,
            default=lambda: orm.Str('relax'),
            help="Option to relax the cell. Possible options are : ['fixed', 'relax', 'vc-relax']")
        #spec.input("qe.dft.unitcell.kpoints", valid_type=orm.KpointsData,
        #    help="The k-point grid to use for the calculations")
        spec.input("qe.dft.unitcell.parameters", valid_type=orm.Dict,
            help="Parameters for the PWSCF calcuations. Some will be set automatically")
        spec.input("qe.dft.unitcell.scheduler_options", valid_type=orm.Dict,
            help="Scheduler options for the PW.x calculations")
        spec.input("qe.dft.unitcell.settings", valid_type=orm.Dict,
            help="Settings for the PW.x calculations")
#        spec.input_namespace("qe.dft.unitcell.pseudopotentials",valid_type=orm.UpfData, dynamic=True,
#            help="The pseudopotential family for use with the code, if required")
        spec.input("qe.dft.unitcell.pseudopotential_family", valid_type=orm.Str,
            help="The pseudopotential family for use with the code")

        # DOS inputs (DOS.x)
        spec.input("qe.dos.code", valid_type=orm.Code,
                help="The dos.x code to use for the calculations")
        spec.input("qe.dos.scheduler_options", valid_type=orm.Dict,
                help="Scheduler options for the dos.x calculations")

        # Postprocessing inputs (PP.x)
        spec.input("qe.pp.code", valid_type=orm.Code,
            help="The pp.x code to use for the calculations")
        spec.input("qe.pp.scheduler_options", valid_type=orm.Dict,
            help="Scheduler options for the PP.x calculations")

        # DFPT inputs (PH.x)
        spec.input("qe.dfpt.code", valid_type=orm.Code, required=False,
            help="The ph.x code to use for the calculations")
        spec.input("qe.dfpt.scheduler_options", valid_type=orm.Dict, required=False,
            help="Scheduler options for the PH.x calculations")

        spec.outline(
            cls.setup,
            if_(cls.if_restart_wc)(
                cls.retrieve_previous_results
                ),
            cls.run_chemical_potential_workchain,
            cls.check_chemical_potential_workchain,
            if_(cls.if_rerun_calc_unitcell)(
                cls.prep_unitcell_dft_calculation,
                cls.check_unitcell_dft_calculation,
                ),
            if_(cls.if_rerun_calc_dos)(
                cls.prep_dos_calculation,
                cls.check_dos_calculation,
                ),
            if_(cls.if_run_dfpt)(
                cls.prep_dfpt_calculation
                ),
            cls.check_dfpt_calculation,
            cls.prep_dft_calculations,
            cls.check_dft_calculations,
            cls.prep_dft_potentials_calculations,
            cls.check_dft_potentials_calculations,
            cls.prep_charge_density_calculations,
            cls.check_charge_density_calculations,
            cls.run_gaussian_correction_workchain,
            cls.check_correction_workchain,
            cls.create_defect_data,
            cls.run_fermi_level_workchain,
            cls.check_fermi_level_workchain,
            cls.compute_defect_formation_energy
            )

    def retrieve_previous_results(self):
        """
        Retrieve all the converged calculations from the previous run
        """

        self.report('Retrieving results from previous calculations...')
        Node = orm.load_node(self.inputs.restart_node.value)

        # Merging and retreiving data from previous run with the that of the additional dopants
        if Node.is_finished_ok:
            self.ctx.chemical_potential = Node.outputs.chemical_potential.get_dict()
            self.ctx.fermi_level = Node.outputs.fermi_level.get_dict()
            self.ctx.total_correction = Node.outputs.total_correction.get_dict()
            self.ctx.electrostatic_correction = Node.outputs.electrostatic_correction.get_dict()
            self.ctx.potential_alignment = Node.outputs.potential_alignment.get_dict()
            # self.ctx.defect_data = Node.outputs.defect_data.get_dict()

            self.ctx.sc_fermi_dopants = list(set(self.ctx.fermi_level.keys()).union(set(self.inputs.formation_energy_dict.get_dict().keys())))

            for defect, properties in self.ctx.all_defects.items():
                # In case we want to add new charge states to the same defects from previous calculations
                # if defect not in self.ctx.defect_data.keys():
                self.ctx.total_correction[defect] = {}
                self.ctx.electrostatic_correction[defect] = {}
                self.ctx.potential_alignment[defect] = {}
                # self.ctx.defect_data[defect] = {'N_site': properties['N_site'], 'species': properties['species'], 'charges': {}}
                if 0.0 not in properties['charges']:
                    self.ctx.all_defects[defect]['charges'].append(0.0)
                # for chg in self.ctx.all_defects[defect]['charges']:
                #     self.ctx.defect_data[defect]['charges'][str(chg)] = {}

            for entry in Node.get_outgoing():
                try:
                    process_label = entry.node.attributes['process_label']
                    #self.report('{}'.format(process_label))
                except KeyError:
                    continue

                #if process_label == 'PwBaseWorkChain':
                #    calc_label = entry.node.label
                #    if 'host' in calc_label:
                #       calc_name = 'calc_'+calc_label
                #       self.report('{}'.format(calc_name))
                #       pw_host_dopants.remove(calc_label[5:])
                #       self.ctx[calc_name] = entry.node

                if process_label == 'FermiLevelWorkchain':
                    self.ctx.dos = entry.node.inputs.DOS
                    vbm = entry.node.inputs.valence_band_maximum.value
                    N_electrons = entry.node.inputs.number_of_electrons.value
                    band_gap = entry.node.inputs.band_gap.value
                    self.ctx['output_unitcell'] = {'number_of_electrons': N_electrons, 'vbm': vbm, 'band_gap': band_gap}

        else:
            for dopant, Ef in Node.inputs.formation_energy_dict.get_dict().items():
                if dopant not in self.ctx.all_dopants.keys():
                    self.ctx.all_dopants[dopant] = Ef
            chempot_dopants = copy.deepcopy(self.ctx.all_dopants)
            sc_fermi_dopants = list(self.ctx.all_dopants.keys()) #copy.deepcopy(self.ctx.all_dopants)
            pw_host_dopants = list(self.ctx.all_dopants.keys())

            for defect, info in Node.inputs.defect_info.get_dict().items():
                if defect not in self.ctx.all_defects.keys():
                    self.ctx.all_defects[defect] = info
                    if 0.0 not in self.ctx.all_defects[defect]['charges']:
                        self.ctx.all_defects[defect]['charges'].append(0.0)
            pw_defects = copy.deepcopy(self.ctx.all_defects)
            phi_defects = copy.deepcopy(self.ctx.all_defects)
            rho_defects = copy.deepcopy(self.ctx.all_defects)
            gc_correction_defects = copy.deepcopy(self.ctx.all_defects)

            for entry in Node.get_outgoing():
                try:
                    process_label = entry.node.attributes['process_label']
                    #self.report('{}'.format(process_label))
                except KeyError:
                    continue

                if process_label == 'PwBaseWorkChain':
                    calc_label = entry.node.label
                    if calc_label == 'unitcell':
                        #calc_name = 'calc_unitcell'
                        #self.ctx['calc_unitcell'] =  entry.node
                        vbm = get_vbm(entry.node)
                        is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(entry.node.outputs.output_band)
                        N_electrons = entry.node.outputs.output_parameters.get_dict()['number_of_electrons']
                        self.ctx['output_unitcell'] = {'number_of_electrons': N_electrons, 'vbm': vbm, 'band_gap': band_gap}
                    elif 'host' in calc_label:
                        calc_name = 'calc_'+calc_label
                        self.report('{}'.format(calc_name))
                        if entry.node.is_finished_ok:
                            pw_host_dopants.remove(calc_label[5:])
                            self.ctx[calc_name] = entry.node
                    else:
                        calc_name = 'calc_defect_'+calc_label
                        if entry.node.is_finished_ok:
                            #self.report('{}'.format(calc_name))
                            self.ctx[calc_name] = entry.node
                            defect, chg = get_defect_and_charge_from_label(calc_label)
                            pw_defects[defect]['charges'].remove(chg)
                            if not pw_defects[defect]['charges']:
                                pw_defects.pop(defect)

                elif process_label == 'PpCalculation':
                    calc_label = entry.node.label
                    if entry.node.is_finished_ok:
                        self.ctx[calc_label] = entry.node
                        if 'host' not in calc_label:
                            defect, chg = get_defect_and_charge_from_label(calc_label[14:])
                            self.report('{}, {}, {}'.format(calc_label, defect, chg))
                            if 'phi' in calc_label:
                            #    self.report('{}'.format(phi_defects))
                                phi_defects[defect]['charges'].remove(chg)
                                if not phi_defects[defect]['charges']:
                                    phi_defects.pop(defect)
                            if 'rho' in calc_label:
                                #self.report('{}'.format(phi_defects))
                                rho_defects[defect]['charges'].remove(chg)
                                if not rho_defects[defect]['charges']:
                                    rho_defects.pop(defect)

                elif process_label == 'DosCalculation':
                    #self.ctx['calc_dos'] = entry.node
                    self.ctx.dos = entry.node.outputs.output_dos

                elif process_label == 'GaussianCounterChargeWorkchain':
                    calc_label = entry.node.label
                    if entry.node.is_finished_ok:
                        self.ctx[calc_label] = entry.node
                        defect, chg = get_defect_and_charge_from_label(calc_label.replace('correction_wc_', ''))
                        gc_correction_defects[defect]['charges'].remove(chg)
                        #if not gc_correction_defects[defect]['charges']:
                        if gc_correction_defects[defect]['charges'] == [0.0]:
                            gc_correction_defects.pop(defect)

                elif process_label == 'ChemicalPotentialWorkchain':
                    dopant = entry.node.label
                    if entry.node.is_finished_ok:
                        self.ctx["chem_potential_wc_{}".format(dopant)] = entry.node
                        chempot_dopants.pop(dopant)

#                elif process_label == 'FermiLevelWorkchain':
#                    dopant = entry.node.label
#                    if entry.node.is_finished_ok:
#                        self.ctx["fermi_level_wc_{}".format(dopant)] = entry.node
#                        sc_fermi_dopants.pop(dopant)

                else:
                    pass

            self.ctx.chempot_dopants = chempot_dopants
            self.ctx.sc_fermi_dopants = sc_fermi_dopants
            self.ctx.pw_host_dopants = pw_host_dopants
            self.ctx.pw_defects = pw_defects
            self.ctx.phi_defects = phi_defects
            self.ctx.rho_defects = rho_defects
            self.ctx.gc_correction_defects = gc_correction_defects

        self.report('chempot dopant: {}'.format(self.ctx.chempot_dopants.keys()))
        self.report('pw host dopant: {}'.format(self.ctx.pw_host_dopants))
        self.report('sc fermi dopants: {}'.format(self.ctx.sc_fermi_dopants))
        self.report('pw defects: {}'.format(self.ctx.pw_defects.keys()))
        self.report('phi defects: {}'.format(self.ctx.phi_defects.keys()))
        self.report('rho defects: {}'.format(self.ctx.rho_defects.keys()))
        self.report('phi defects: {}'.format(self.ctx.phi_defects.keys()))
        self.report('rho defects: {}'.format(self.ctx.rho_defects.keys()))
        self.report('gc correction defects: {}'.format(self.ctx.gc_correction_defects.keys()))

    def prep_unitcell_dft_calculation(self):
        """
        Run a DFT calculation on the structure to be used for the computation of the
        DOS and/or dielectric constant
        """
        self.report("DFT calculation for the unitcell has been requested")

        # Another code may be desirable - N.B. in AiiDA a code refers to a specific
        # executable on a specific computer. As the PH calculation may have to be run on
        # an HPC cluster, the PW calculation must be run on the same machine and so this
        # may necessitate that a different code is used than that for the supercell calculations.

        relax_type = {'fixed': RelaxType.NONE, 'relax': RelaxType.POSITIONS, 'vc-relax': RelaxType.POSITIONS_CELL}
        overrides = {
                'base':{
                    # 'pseudo_family': self.inputs.qe.dft.unitcell.pseudopotential_family.value,
                    'pw': {}
                    },
                'base_final_scf':{
                    # 'pseudo_family': self.inputs.qe.dft.unitcell.pseudopotential_family.value,
                    'pw': {}
                    },
                'clean_workdir' : orm.Bool(False),
                }

        if 'pseudopotential_family' in self.inputs.qe.dft.unitcell:
            overrides['base']['pseudo_family'] = self.inputs.qe.dft.unitcell.pseudopotential_family.value
            overrides['base_final_scf']['pseudo_family'] = self.inputs.qe.dft.unitcell.pseudopotential_family.value
        if 'parameters' in self.inputs.qe.dft.unitcell:
            overrides['base']['pw']['parameters'] = self.inputs.qe.dft.unitcell.parameters.get_dict()
            overrides['base_final_scf']['pw']['parameters'] = self.inputs.qe.dft.unitcell.parameters.get_dict()
        if 'scheduler_options' in self.inputs.qe.dft.unitcell:
            overrides['base']['pw']['metadata'] = self.inputs.qe.dft.unitcell.scheduler_options.get_dict()
            overrides['base_final_scf']['pw']['metadata'] = self.inputs.qe.dft.unitcell.scheduler_options.get_dict()
        if 'settings' in self.inputs.qe.dft.unitcell:
            overrides['base']['pw']['settings'] = self.inputs.qe.dft.unitcell.settings.get_dict()
            overrides['base_final_scf']['pw']['settings'] = self.inputs.qe.dft.unitcell.settings.get_dict()

        inputs = PwRelaxWorkChain.get_builder_from_protocol(
                    code = self.inputs.qe.dft.unitcell.code,
                    structure = self.inputs.unitcell,
                    overrides = overrides,
                    relax_type = relax_type[self.inputs.qe.dft.unitcell.relaxation_scheme.value]
                    )

        future = self.submit(inputs)
        self.report(
            'Launching PWSCF for the unitcell structure (PK={}) at node PK={}'.format(self.inputs.unitcell.pk, future.pk))
        future.label = 'unitcell'
        self.to_context(**{'calc_unitcell': future})

    def check_unitcell_dft_calculation(self):
        """
        Check if the DFT calculation of the unitcell has completed successfully.
        """

        # self.ctx['calc_unitcell'] = orm.load_node(230976)

        unitcell_calc = self.ctx['calc_unitcell']
        if not unitcell_calc.is_finished_ok:
            self.report(
                'PWSCF for the unitcell structure has failed with status {}'.
                format(unitcell_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED
        else:
            is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(unitcell_calc.outputs.output_band)
            if not is_insulator:
                self.report('WARNING! Metallic ground state!')
            self.ctx.vbm = orm.Float(get_vbm(unitcell_calc))
            #self.ctx.number_of_electrons = unitcell_calc.res.number_of_electrons
            self.ctx.number_of_electrons = unitcell_calc.outputs.output_parameters.get_dict()['number_of_electrons']
            self.ctx.band_gap = band_gap
            self.report("The band gap of the material is: {} eV".format(band_gap))
            self.report("The number of electron is: {}".format(self.ctx.number_of_electrons))
            self.report("The bottom of the valence band is: {} eV".format(self.ctx.vbm.value))


    def prep_dos_calculation(self):
        '''
        Run a calculation to extract the DOS of the unitcell.
        '''
        dos_inputs = DosCalculation.get_builder()
        dos_inputs.code = self.inputs.qe.dos.code
        dos_inputs.parent_folder = self.ctx['calc_unitcell'].outputs.remote_folder

        parameters = orm.Dict(dict={'DOS':{
            'Emin': -180.0, 'Emax': 40.0, 'degauss':0.0005, 'DeltaE': 0.005}
            })
        dos_inputs.parameters = parameters

        dos_inputs.metadata = self.inputs.qe.dos.scheduler_options.get_dict()

        future = self.submit(DosCalculation, **dos_inputs)
        self.report('Launching DOS for unitcell structure (PK={}) at node PK={}'.format(self.inputs.unitcell.pk, future.pk))
        self.to_context(**{'calc_dos': future})

    def check_dos_calculation(self):
        '''
        Retrieving the DOS of the unitcell
        '''

        # self.ctx['calc_dos'] = orm.load_node(230991)

        dos_calc = self.ctx['calc_dos']
        if dos_calc.is_finished_ok:
            Dos = dos_calc.outputs.output_dos
            x = Dos.get_x()
            y = Dos.get_y()
            DOS = np.vstack((x[1]-self.ctx.vbm.value, y[1][1])).T
            mask = (DOS[:,0] <= 0.05)
            N_electron = np.trapz(DOS[:,1][mask], DOS[:,0][mask])
            if np.absolute(N_electron - self.ctx.number_of_electrons) > 5e-3:
                self.report('The number of electrons obtained from the integration of DOS is: {}'.format(N_electron))
                self.report('The number of electrons obtained from the integration of DOS is different from the expected number of electrons in the input')
                return self.exit_codes.ERROR_DOS_INTEGRATION_FAILED
            else:
                self.ctx.dos = dos_calc.outputs.output_dos
        else:
            self.report('DOS calculation for the unitcell has failed with status {}'.format(dos_calc.exit_status))
            return self.exit_codes.ERROR_DOS_CALCULATION_FAILED

    def prep_dfpt_calculation(self):
        """
        Run a DFPT calculation to compute the dielectric constant for the pristine material
        """

        ph_inputs = PhCalculation.get_builder()
        ph_inputs.code = self.inputs.qe.dfpt.code

        # Setting up the calculation depends on whether the parent SCF calculation is either
        # the host supercell or an alternative host unitcell
        if self.inputs.unitcell:
            ph_inputs.parent_folder = self.ctx['calc_unitcell'].outputs.remote_folder
        else:
            ph_inputs.parent_folder = self.ctx['calc_host'].outputs.remote_folder

        parameters = orm.Dict(dict={
            'INPUTPH': {
                "tr2_ph" : 1e-16,
                'epsil': True,
                'trans': False
            }
        })
        ph_inputs.parameters = parameters

        # Set the q-points for a Gamma-point calculation
        # N.B. Setting a 1x1x1 mesh is not equivalent as this will trigger a full phonon dispersion calculation
        qpoints = orm.KpointsData()
        if self.inputs.host_unitcell:
            qpoints.set_cell_from_structure(structuredata=self.ctx['calc_unitcell'].inputs.structure)
        else:
            qpoints.set_cell_from_structure(structuredata=self.ctx['calc_host'].inputs.structure)
        qpoints.set_kpoints([[0.,0.,0.]])
        qpoints.get_kpoints(cartesian=True)
        ph_inputs.qpoints = qpoints

        ph_inputs.metadata = self.inputs.qe.dfpt.scheduler_options.get_dict()

        future = self.submit(PhCalculation, **ph_inputs)
        self.report('Launching PH for host structure at node PK={}'.format(self.inputs.host_structure.pk, future.pk))
        self.to_context(**{'calc_dfpt': future})

    def check_dfpt_calculation(self):
        """
        Compute the dielectric constant to be used in the correction
        """
        if self.inputs.epsilon == 0.0:
            dfpt_calc = self.ctx['calc_dfpt']
            if dfpt_calc.is_finished_ok:
                epsilion_tensor = np.array(dfpt_calc.outputs.output_parameters.get_dict()['dielectric_constant'])
                self.ctx.epsilon = orm.Float(np.trace(epsilion_tensor/3.))
                self.report('The computed relative permittivity is {}'.format(self.ctx.epsilon.value))
            else:
                self.report(
                    'PH for the host structure has failed with status {}'.format(dfpt_calc.exit_status))
                return self.exit_codes.ERROR_DFPT_CALCULATION_FAILED
        else:
            self.ctx.epsilon = self.inputs.epsilon

    def prep_dft_calculations(self):
        """
        Run DFT calculation of the perfect host lattice as well as all the possible defects considered in the material.
        """

        relax_type = {'fixed': RelaxType.NONE, 'relax': RelaxType.POSITIONS, 'vc-relax': RelaxType.POSITIONS_CELL}
        overrides = {
                'base':{
                    # 'pseudo_family': self.inputs.qe.dft.unitcell.pseudopotential_family.value,
                    'pw': {}
                    },
                'base_final_scf':{
                    # 'pseudo_family': self.inputs.qe.dft.unitcell.pseudopotential_family.value,
                    'pw': {}
                    },
                'clean_workdir' : orm.Bool(False),
                }

        if 'pseudopotential_family' in self.inputs.qe.dft.supercell:
            overrides['base']['pseudo_family'] = self.inputs.qe.dft.supercell.pseudopotential_family.value
            overrides['base_final_scf']['pseudo_family'] = self.inputs.qe.dft.supercell.pseudopotential_family.value
        if 'parameters' in self.inputs.qe.dft.supercell:
            overrides['base']['pw']['parameters'] = self.inputs.qe.dft.supercell.parameters.get_dict()
            overrides['base_final_scf']['pw']['parameters'] = self.inputs.qe.dft.supercell.parameters.get_dict()
        if 'scheduler_options' in self.inputs.qe.dft.supercell:
            overrides['base']['pw']['metadata'] = self.inputs.qe.dft.supercell.scheduler_options.get_dict()
            overrides['base_final_scf']['pw']['metadata'] = self.inputs.qe.dft.supercell.scheduler_options.get_dict()
        if 'settings' in self.inputs.qe.dft.supercell:
            overrides['base']['pw']['settings'] = self.inputs.qe.dft.supercell.settings.get_dict()
            overrides['base_final_scf']['pw']['settings'] = self.inputs.qe.dft.supercell.settings.get_dict()

        for dopant in self.ctx.pw_host_dopants:
        #for dopant in self.ctx.pw_host_dopants[:1]:
            #overrides['base']['pw']['metadata']['label'] = 'host_{}'.format(dopant)
            inputs = PwRelaxWorkChain.get_builder_from_protocol(
                    code = self.inputs.qe.dft.supercell.code,
                    structure = self.inputs.host_structure,
                    overrides = overrides,
                    relax_type = relax_type[self.inputs.qe.dft.supercell.relaxation_scheme.value]
                    )
            future = self.submit(inputs)
            self.report(
                'Launching PWSCF for host structure (PK={}) for {} dopant at node PK={}'.format(self.inputs.host_structure.pk, dopant, future.pk))
            future.label = 'host_{}'.format(dopant)
            self.to_context(**{'calc_host_{}'.format(dopant): future})


        for defect, properties in self.ctx.pw_defects.items():
            defect_structure = generate_defect_structure(self.inputs.host_structure, properties['defect_position'], properties['species'])
            for chg in properties['charges']:
                overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'SYSTEM':{'tot_charge': chg}})
                overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'SYSTEM':{'tot_charge': chg}})

                inputs = PwRelaxWorkChain.get_builder_from_protocol(
                        code = self.inputs.qe.dft.supercell.code,
                        structure = defect_structure,
                        overrides = overrides,
                        relax_type = relax_type[self.inputs.qe.dft.supercell.relaxation_scheme.value]
                        )

                future = self.submit(inputs)
                self.report('Launching PWSCF for {} defect structure with charge {} at node PK={}'
                        .format(defect, chg, future.pk))
                future.label = '{}[{}]'.format(defect, chg)
                self.to_context(**{'calc_defect_{}[{}]'.format(defect, chg): future})

#    def prep_dft_calculations(self):
#        """
#        Run DFT calculation of the perfect host lattice as well as all the possible defects considered in the material.
#        """
#
##        pw_inputs = PwCalculation.get_builder()
##        pw_inputs.code = self.inputs.qe.dft.supercell.code
#
#        kpoints = orm.KpointsData()
#        kpoints.set_cell_from_structure(self.inputs.host_structure)
#        kpoints.set_kpoints_mesh_from_density(self.inputs.k_points_distance.value)
##        pw_inputs.kpoints = kpoints
#
##        pw_inputs.metadata = self.inputs.qe.dft.supercell.scheduler_options.get_dict()
##        pw_inputs.settings = self.inputs.qe.dft.supercell.settings
#        scheduler_options = self.inputs.qe.dft.supercell.scheduler_options.get_dict()
#        parameters = self.inputs.qe.dft.supercell.parameters.get_dict()
#
#        # We set 'tot_charge' later so throw an error if the user tries to set it to avoid
#        # any ambiguity or unseen modification of user input
#        if 'tot_charge' in parameters['SYSTEM']:
#            self.report('You cannot set the "tot_charge" PW.x parameter explicitly')
#            return self.exit_codes.ERROR_PARAMETER_OVERRIDE
#
##        pw_inputs.structure = self.inputs.host_structure
#        parameters['SYSTEM']['tot_charge'] = orm.Float(0.)
##        pw_inputs.parameters = orm.Dict(dict=parameters)
#
#        inputs = {
#            'pw':{
#                'code' : self.inputs.qe.dft.supercell.code,
#                'structure' : self.inputs.host_structure,
#                'parameters' : orm.Dict(dict=parameters),
#                'settings': self.inputs.qe.dft.supercell.settings,
#            },
#            'kpoints': kpoints,
#        }
#
#        for dopant in self.ctx.pw_host_dopants[:1]:
#            pseudos = get_pseudos_from_structure(self.inputs.host_structure, self.inputs.qe.dft.supercell.pseudopotential_family.value)
#            scheduler_options['label'] = 'host_{}'.format(dopant)
##            pw_inputs.metadata = scheduler_options
#
#            inputs['pw']['pseudos'] = pseudos
#            inputs['pw']['metadata'] = scheduler_options
#
#            future = self.submit(PwBaseWorkChain, **inputs)
#            self.report('Launching PWSCF for host structure (PK={}) at node PK={}'
#                    .format(self.inputs.host_structure.pk, future.pk))
#            future.label = 'host_{}'.format(dopant)
#            self.to_context(**{'calc_host_{}'.format(dopant): future})
#
#        #defect_info = self.inputs.defect_info.get_dict()
#        for defect, properties in self.ctx.pw_defects.items():
#            defect_structure = generate_defect_structure(self.inputs.host_structure, properties['defect_position'], properties['species'])
##            temp_structure = pymatgen.Structure.from_file('/home/sokseiham/Documents/Defect_calculations/LiK2AlF6/Structures/Ba-K.cif')
##            defect_structure = orm.StructureData(pymatgen=temp_structure)
#            pseudos = get_pseudos_from_structure(defect_structure, self.inputs.qe.dft.supercell.pseudopotential_family.value)
#
#            inputs['pw']['structure'] = defect_structure
#            inputs['pw']['pseudos'] = pseudos
#
#            parameters['SYSTEM']['nspin'] = 2
#            parameters['SYSTEM']['tot_magnetization'] = 0.0
#
#            for chg in properties['charges']:
#                parameters['SYSTEM']['tot_charge'] = orm.Float(chg)
##                pw_inputs.parameters = orm.Dict(dict=parameters)
#                scheduler_options['label'] = '{}[{}]'.format(defect, chg)
##                pw_inputs.metadata = scheduler_options
#
#                inputs['pw']['metadata'] = scheduler_options
#                inputs['pw']['parameters'] = orm.Dict(dict=parameters)
#
#                future = self.submit(PwBaseWorkChain, **inputs)
#                self.report('Launching PWSCF for {} defect structure with charge {} at node PK={}'
#                        .format(defect, chg, future.pk))
#                future.label = '{}[{}]'.format(defect, chg)
#                self.to_context(**{'calc_defect_{}[{}]'.format(defect, chg): future})

    def check_dft_calculations(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """

        # self.ctx['calc_host_intrinsic'] = orm.load_node(231011)
        # self.ctx['calc_defect_N-O[-1.0]'] = orm.load_node(231028)
        # self.ctx['calc_defect_N-O[0.0]'] = orm.load_node(231044)
        # self.ctx['calc_defect_V_Cl[1.0]'] = orm.load_node(231061)
        # self.ctx['calc_defect_V_Cl[0.0]'] = orm.load_node(231077)

        # Host
        for dopant in self.ctx.pw_host_dopants[:1]:
            host_calc = self.ctx['calc_host_{}'.format(dopant)]
#            if host_calc.is_finished_ok:
#                self.ctx.host_energy = orm.Float(host_calc.outputs.output_parameters.get_dict()['energy']) # eV
#                self.report('The energy of the host is: {} eV'.format(self.ctx.host_energy.value))
#                self.ctx.host_vbm = orm.Float(get_vbm(host_calc))
#                self.report('The top of valence band is: {} eV'.format(self.ctx.host_vbm.value))
            if not host_calc.is_finished_ok:
                self.report(
                    'PWSCF for the host structure has failed with status {}'.format(host_calc.exit_status))
                return self.exit_codes.ERROR_DFT_CALCULATION_FAILED

        # Defects
        #defect_info = self.inputs.defect_info.get_dict()
        defect_info = self.ctx.all_defects
        for defect, properties in defect_info.items():
            dopant = get_dopant(properties['species'], self.inputs.compound.value)
            #self.ctx.defect_data[defect]['vbm'] = get_vbm(self.ctx['calc_host_{}'.format(dopant)])
            #self.ctx.defect_data[defect]['E_host'] = self.ctx['calc_host_{}'.format(dopant)].outputs.output_parameters.get_dict()['energy']
            # if self.ctx.pw_host_dopants == []:
            #     self.ctx.defect_data[defect]['vbm'] = get_vbm(self.ctx['calc_host_intrinsic'])
            #     self.ctx.defect_data[defect]['E_host'] = self.ctx['calc_host_intrinsic'].outputs.output_parameters.get_dict()['energy']
            # else:
            #     self.ctx.defect_data[defect]['vbm'] = get_vbm(self.ctx['calc_host_{}'.format(self.ctx.pw_host_dopants[0])])
            #     self.ctx.defect_data[defect]['E_host'] = self.ctx['calc_host_{}'.format(self.ctx.pw_host_dopants[0])].outputs.output_parameters.get_dict()['energy']
            for chg in properties['charges']:
                defect_calc = self.ctx['calc_defect_{}[{}]'.format(defect, chg)]
                if not defect_calc.is_finished_ok:
                    self.report('PWSCF for the {} defect structure with charge {} has failed with status {}'
                            .format(defect, chg, defect_calc.exit_status))
                    return self.exit_codes.ERROR_DFT_CALCULATION_FAILED
                else:
                    is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(defect_calc.outputs.output_band)
                    if not is_insulator:
                        self.report('WARNING! The ground state of {} defect structure with charge {} is metallic!'.format(defect, chg))
                    # self.ctx.defect_data[defect]['charges'][str(chg)]['E'] = defect_calc.outputs.output_parameters.get_dict()['energy'] # eV
                    self.report('The energy of {} defect structure with charge {} is: {} eV'
                            .format(defect, chg, defect_calc.outputs.output_parameters.get_dict()['energy']))
#        self.report('The defect data is :{}'.format(self.ctx.defect_data))

    def prep_dft_potentials_calculations(self):
        """
        Obtain the electrostatic potentials from the PWSCF calculations.
        """
        # User inputs
        pp_inputs = PpCalculation.get_builder()
        pp_inputs.code = self.inputs.qe.pp.code

        scheduler_options = self.inputs.qe.pp.scheduler_options.get_dict()
        scheduler_options['label'] = 'pp_phi_host'
        pp_inputs.metadata = scheduler_options

        # Fixed settings
        #pp_inputs.plot_number = orm.Int(11)  # Electrostatic potential
        #pp_inputs.plot_dimension = orm.Int(3)  # 3D

        parameters = orm.Dict(dict={
            'INPUTPP': {
                "plot_num" : 11,
            },
            'PLOT': {
                "iflag" : 3
            }
        })
        pp_inputs.parameters = parameters

        # Host
        # assuming that the electrostatic potential doesnt vary much with the cutoff
        # pp_inputs.parent_folder = self.ctx['calc_host_intrinsic'].outputs.remote_folder
        self.report('pw_host_dopants: {}'.format(self.ctx.pw_host_dopants))
        if self.ctx.pw_host_dopants == []:
            pp_inputs.parent_folder = self.ctx['calc_host_intrinsic'].outputs.remote_folder
        else:
            pp_inputs.parent_folder = self.ctx['calc_host_{}'.format(self.ctx.pw_host_dopants[0])].outputs.remote_folder
        future = self.submit(PpCalculation, **pp_inputs)
        self.report('Launching PP.x for electrostatic potential for the host structure at node PK={}'
                .format(future.pk))
        self.to_context(**{'pp_phi_host': future})

        #Defects
        for defect, properties in self.ctx.phi_defects.items():
            for chg in properties['charges']:
                scheduler_options['label'] = 'pp_phi_defect_{}[{}]'.format(defect, chg)
                pp_inputs.metadata = scheduler_options
                pp_inputs.parent_folder = self.ctx['calc_defect_{}[{}]'.format(defect, chg)].outputs.remote_folder
                future = self.submit(PpCalculation, **pp_inputs)
                self.report('Launching PP.x for electrostatic potential for {} defect structure with charge {} at node PK={}'
                        .format(defect, chg, future.pk))
                self.to_context(**{'pp_phi_defect_{}[{}]'.format(defect, chg): future})

    def check_dft_potentials_calculations(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """

        # self.ctx['pp_phi_host'] = orm.load_node(231144)
        # self.ctx['pp_phi_defect_N-O[-1.0]'] = orm.load_node(231145)
        # self.ctx['pp_phi_defect_N-O[0.0]'] = orm.load_node(231146)
        # self.ctx['pp_phi_defect_V_Cl[1.0]'] = orm.load_node(231147)
        # self.ctx['pp_phi_defect_V_Cl[0.0]'] = orm.load_node(231148)

        # Host
        host_pp = self.ctx['pp_phi_host']
        if host_pp.is_finished_ok:
            #data_array = host_pp.outputs.output_data.get_array('data')
            #v_data = orm.ArrayData()
            #v_data.set_array('data', data_array)
            #self.ctx.phi_host = v_data
            self.ctx.phi_host = get_data_array(host_pp.outputs.output_data)
        else:
            self.report(
                'Post processing for electrostatic potential the host structure has failed with status {}'.format(host_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED

        # Defects
        defect_info = self.ctx.all_defects
        for defect, properties in defect_info.items():
            for chg in properties['charges']:
                defect_pp = self.ctx['pp_phi_defect_{}[{}]'.format(defect, chg)]
                if defect_pp.is_finished_ok:
                    #data_array = defect_pp.outputs.output_data.get_array('data')
                    #v_data = orm.ArrayData()
                    #v_data.set_array('data', data_array)
                    #self.ctx['phi_defect_{}[{}]'.format(defect, chg)] = v_data
                    self.ctx['phi_defect_{}[{}]'.format(defect, chg)] = get_data_array(defect_pp.outputs.output_data)
                else:
                    self.report('Post processing for electrostatic potential for {} defect structure with charge {} has failed with status {}'
                            .format(defect, chg, defect_pp.exit_status))
                    return self.exit_codes.ERROR_PP_CALCULATION_FAILED

    def prep_charge_density_calculations(self):
        """
        Obtain electronic charge density from the PWSCF calculations.
        """
        # User inputs
        pp_inputs = PpCalculation.get_builder()
        pp_inputs.code = self.inputs.qe.pp.code
        scheduler_options = self.inputs.qe.pp.scheduler_options.get_dict()
        scheduler_options['label'] = 'pp_rho_host'
        pp_inputs.metadata = scheduler_options

        # Fixed settings
        #pp_inputs.plot_number = orm.Int(0)  # Charge density
        #pp_inputs.plot_dimension = orm.Int(3)  # 3D

        parameters = orm.Dict(dict={
            'INPUTPP': {
                "plot_num" : 0,
            },
            'PLOT': {
                "iflag" : 3
            }
        })
        pp_inputs.parameters = parameters

        # Host
        # assuming that the charge density doesn't vary much with the cutoff
        pp_inputs.parent_folder = self.ctx['calc_host_{}'.format(self.ctx.pw_host_dopants[0])].outputs.remote_folder
        #pp_inputs.parent_folder = self.ctx['calc_host_intrinsic'].outputs.remote_folder

        future = self.submit(PpCalculation, **pp_inputs)
        self.report('Launching PP.x for charge density for host structure at node PK={}'
                .format(future.pk))
        self.to_context(**{'pp_rho_host': future})

        #Defects
        for defect, properties in self.ctx.rho_defects.items():
            for chg in properties['charges']:
                pp_inputs.parent_folder = self.ctx['calc_defect_{}[{}]'.format(defect, chg)].outputs.remote_folder
                scheduler_options['label'] = 'pp_rho_defect_{}[{}]'.format(defect, chg)
                pp_inputs.metadata = scheduler_options
                future = self.submit(PpCalculation, **pp_inputs)
                self.report('Launching PP.x for charge density for {} defect structure with charge {} at node PK={}'
                        .format(defect, chg, future.pk))
                self.to_context(**{'pp_rho_defect_{}[{}]'.format(defect, chg): future})

    def check_charge_density_calculations(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """

        # self.ctx['pp_rho_host'] = orm.load_node(231180)
        # self.ctx['pp_rho_defect_N-O[-1.0]'] = orm.load_node(231181)
        # self.ctx['pp_rho_defect_N-O[0.0]'] = orm.load_node(231182)
        # self.ctx['pp_rho_defect_V_Cl[1.0]'] = orm.load_node(231183)
        # self.ctx['pp_rho_defect_V_Cl[0.0]'] = orm.load_node(231184)

        # Host
        host_pp = self.ctx['pp_rho_host']
        if host_pp.is_finished_ok:
            #data_array = host_pp.outputs.output_data.get_array('data')
            #v_data = orm.ArrayData()
            #v_data.set_array('data', data_array)
            #self.ctx.rho_host = v_data
            self.ctx.rho_host = get_data_array(host_pp.outputs.output_data)
        else:
            self.report(
                'Post processing for charge density for the host structure has failed with status {}'.format(host_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED

        # Defects
        defect_info = self.ctx.all_defects
        for defect, properties in defect_info.items():
            for chg in properties['charges']:
                defect_pp = self.ctx['pp_rho_defect_{}[{}]'.format(defect, chg)]
                if defect_pp.is_finished_ok:
                    #data_array = defect_pp.outputs.output_data.get_array('data')
                    #v_data = orm.ArrayData()
                    #v_data.set_array('data', data_array)
                    #self.ctx['rho_defect_{}[{}]'.format(defect, chg)] = v_data
                    self.ctx['rho_defect_{}[{}]'.format(defect, chg)] = get_data_array(defect_pp.outputs.output_data)
                else:
                    self.report('Post processing for charge density for {} defect structure with charge {} has failed with status {}'
                            .format(defect, chg, defect_pp.exit_status))
                    return self.exit_codes.ERROR_PP_CALCULATION_FAILED

