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
from aiida.engine import WorkChain, calcfunction, ToContext, if_, submit
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_quantumespresso.common.types import RelaxType

from aiida_defects.formation_energy.formation_energy_base import FormationEnergyWorkchainBase
from aiida_defects.formation_energy.utils import run_pw_calculation
from .utils import get_vbm, get_raw_formation_energy, get_data_array, get_corrected_formation_energy, get_corrected_aligned_formation_energy

PpCalculation = CalculationFactory('quantumespresso.pp')

class FormationEnergyWorkchainQE(FormationEnergyWorkchainBase):
    """
    Compute the formation energy for a given defect using QuantumESPRESSO
    """
    @classmethod
    def define(cls, spec):
        super(FormationEnergyWorkchainQE, cls).define(spec)

        # DFT and DFPT calculations with QuantumESPRESSO are handled with different codes, so here
        # we keep track of things with two separate namespaces. An additional code, and an additional
        # namespace, is used for postprocessing
        spec.input_namespace('qe.dft.supercell',
            help="Inputs for DFT calculations on supercells")
        spec.input_namespace('qe.dft.unitcell', required=False,
            help="Inputs for a DFT calculation on an alternative host cell for use with DFPT")
        spec.input_namespace('qe.dfpt',
            help="Inputs for DFPT calculation for calculating the relative permittivity of the host material")
        spec.input_namespace('qe.pp',
            help="Inputs for postprocessing calculations")


        # What calculations to run
        spec.input('run_pw_host', valid_type=orm.Bool, required=True)  # TODO: Check why these are here - for restarts?
        spec.input('run_pw_defect_q0', valid_type=orm.Bool, required=True)
        spec.input('run_pw_defect_q', valid_type=orm.Bool, required=True)
        spec.input('run_v_host', valid_type=orm.Bool, required=True)
        spec.input('run_v_defect_q0', valid_type=orm.Bool, required=True)
        spec.input('run_v_defect_q', valid_type=orm.Bool, required=True)
        spec.input('run_rho_host', valid_type=orm.Bool, required=True)
        spec.input('run_rho_defect_q0', valid_type=orm.Bool, required=True)
        spec.input('run_rho_defect_q', valid_type=orm.Bool, required=True)
        spec.input('run_dfpt', valid_type=orm.Bool, required=True)

        spec.input('host_node', valid_type=orm.Int, required=False)  # TODO: Need to look at this if this is intended for passing parent calcs
        spec.input('defect_q0_node', valid_type=orm.Int, required=False)
        spec.input('defect_q_node', valid_type=orm.Int, required=False)
        spec.input('v_host_node', valid_type=orm.Int, required=False)
        spec.input('v_defect_q0_node', valid_type=orm.Int, required=False)
        spec.input('v_defect_q_node', valid_type=orm.Int, required=False)
        spec.input('rho_host_node', valid_type=orm.Int, required=False)
        spec.input('rho_defect_q0_node', valid_type=orm.Int, required=False)
        spec.input('rho_defect_q_node', valid_type=orm.Int, required=False)
        spec.input("relaxation_scheme", valid_type=orm.Str, required=False,
            default=lambda: orm.Str('vc-relax'),
            help="Option to relax the cell. Possible options are : ['fixed', 'relax', 'vc-relax']")

        # DFT inputs (PW.x)
        spec.input("qe.dft.supercell.code", valid_type=orm.Code,
            help="The pw.x code to use for the calculations")
        spec.input("qe.dft.supercell.parameters", valid_type=orm.Dict, required=False,
            help="Parameters for the PWSCF calcuations. Some will be set automatically")
        spec.input("qe.dft.supercell.scheduler_options", valid_type=orm.Dict,
            help="Scheduler options for the PW.x calculations")
        spec.input("qe.dft.supercell.settings", valid_type=orm.Dict,
            help="Settings for the PW.x calculations")
        spec.input("qe.dft.supercell.pseudopotential_family", valid_type=orm.Str,
            help="The pseudopotential family for use with the code")

        # DFT inputs (PW.x) for the unitcell calculation for the dielectric constant
        spec.input("qe.dft.unitcell.code", valid_type=orm.Code,
            help="The pw.x code to use for the calculations")
        spec.input("qe.dft.unitcell.parameters",
            valid_type=orm.Dict, required=False,
            help="Parameters for the PWSCF calcuations. Some will be set automatically")
        spec.input("qe.dft.unitcell.scheduler_options",
            valid_type=orm.Dict,
            help="Scheduler options for the PW.x calculations")
        spec.input("qe.dft.unitcell.settings", valid_type=orm.Dict,
            help="Settings for the PW.x calculations")
        spec.input("qe.dft.unitcell.pseudopotential_family", valid_type=orm.Str,
            help="The pseudopotential family for use with the code")

        # Postprocessing inputs (PP.x)
        spec.input("qe.pp.code",
            valid_type=orm.Code,
            help="The pp.x code to use for the calculations")
        spec.input("qe.pp.scheduler_options",
            valid_type=orm.Dict,
            help="Scheduler options for the PP.x calculations")

        # DFPT inputs (PH.x)
        spec.input("qe.dfpt.code",
            valid_type=orm.Code,
            help="The ph.x code to use for the calculations")
        spec.input("qe.dfpt.scheduler_options",
            valid_type=orm.Dict,
            help="Scheduler options for the PH.x calculations")

        spec.outline(
            cls.setup,
            if_(cls.if_run_chem_pot_wc)(
                cls.run_chemical_potential_workchain,
            ),
            cls.check_chemical_potential_workchain,
            if_(cls.correction_required)(
                if_(cls.is_gaussian_scheme)(
                    cls.prep_dft_calcs_gaussian_correction,
                    cls.check_dft_calcs_gaussian_correction,
                    cls.get_dft_potentials_gaussian_correction,
                    cls.check_dft_potentials_gaussian_correction,
                    cls.get_charge_density,
                    cls.check_charge_density_calculations,
                    if_(cls.if_run_dfpt)(
                        cls.prep_hostcell_calc_for_dfpt,
                        cls.check_hostcell_calc_for_dfpt,
                        cls.prep_calc_dfpt_calculation,
                    ),
                    cls.get_permittivity,
                    cls.run_gaussian_correction_workchain),
                if_(cls.is_point_scheme)(
                    cls.raise_not_implemented
                    #cls.prepare_point_correction_workchain,
                    #cls.run_point_correction_workchain),
                ),
                cls.check_correction_workchain),
            cls.compute_formation_energy
        )

    def prep_dft_calcs_gaussian_correction(self):
        """
        Get the required inputs for the Gaussian Countercharge correction workchain.
        This method runs the required calculations to generate the energies and potentials
        for the Gaussian scheme.
        """

        self.report("Setting up the Gaussian Countercharge correction workchain")

        relax_type = {'fixed': RelaxType.NONE, 'relax': RelaxType.POSITIONS, 'vc-relax': RelaxType.POSITIONS_CELL}

        overrides = {
                'base':{
                    # 'pseudo_family': self.inputs.qe.dft.supercell.pseudopotential_family.value,
                    'pw': {
                    	'parameters': {},
                        # 'metadata': self.inputs.qe.dft.supercell.scheduler_options.get_dict(),
                        'settings': self.inputs.qe.dft.supercell.settings.get_dict(),
                        }
                    },
                'base_final_scf':{
                    # 'pseudo_family': self.inputs.qe.dft.supercell.pseudopotential_family.value,
                    'pw': {
                    	'parameters': {},
                        # 'metadata': self.inputs.qe.dft.supercell.scheduler_options.get_dict(),
                        'settings': self.inputs.qe.dft.supercell.settings.get_dict(),
                        }
                    },
                'clean_workdir' : orm.Bool(False),
                }

        if 'pseudopotential_family' in self.inputs.qe.dft.supercell:
        	overrides['base']['pseudo_family'] = self.inputs.qe.dft.supercell.pseudopotential_family.value
        	overrides['base_final_scf']['pseudo_family'] = self.inputs.qe.dft.supercell.pseudopotential_family.value
        if 'parameters' in self.inputs.qe.dft.supercell:
            overrides['base']['pw']['parameters'] = self.inputs.qe.dft.supercell.parameters.get_dict()
            overrides['base_final_scf']['pw']['parameters'] = self.inputs.qe.dft.supercell.parameters.get_dict()
        # else:
        #     overrides['base']['pw']['parameters'] = {}
        #     overrides['base_final_scf']['pw']['parameters'] = {}

        # Host structure
        if self.inputs.run_pw_host:
            inputs = PwRelaxWorkChain.get_builder_from_protocol(
                    code = self.inputs.qe.dft.supercell.code,
                    structure = self.inputs.host_structure,
                    overrides = overrides,
                    relax_type = relax_type[self.inputs.relaxation_scheme.value]
                    )

            inputs['base']['pw']['metadata'] = self.inputs.qe.dft.supercell.scheduler_options.get_dict()
            inputs['base']['pw']['settings'] = self.inputs.qe.dft.supercell.settings
            inputs['base_final_scf']['pw']['metadata'] = self.inputs.qe.dft.supercell.scheduler_options.get_dict()
            inputs['base_final_scf']['pw']['settings'] = self.inputs.qe.dft.supercell.settings

            #future = self.submit(PwRelaxWorkChain, **inputs)
            future = self.submit(inputs)
            self.report(
                'Launching PWSCF for the host structure (PK={}) with charge {} (PK={})'
                .format(self.inputs.host_structure.pk, "0.0", future.pk))
            self.to_context(**{'calc_host': future})

        # Defect structure; neutral charge state
        if self.inputs.run_pw_defect_q0:
            inputs = PwRelaxWorkChain.get_builder_from_protocol(
                    code = self.inputs.qe.dft.supercell.code,
                    structure = self.inputs.defect_structure,
                    overrides = overrides,
                    relax_type = relax_type[self.inputs.relaxation_scheme.value]
                    )

            inputs['base']['pw']['metadata'] = self.inputs.qe.dft.supercell.scheduler_options.get_dict()
            inputs['base']['pw']['settings'] = self.inputs.qe.dft.supercell.settings
            inputs['base_final_scf']['pw']['metadata'] = self.inputs.qe.dft.supercell.scheduler_options.get_dict()
            inputs['base_final_scf']['pw']['settings'] = self.inputs.qe.dft.supercell.settings

            #future = self.submit(PwRelaxWorkChain, **inputs)
            future = self.submit(inputs)
            self.report(
                'Launching PWSCF for the defect structure (PK={}) with charge {} (PK={})'
                .format(self.inputs.defect_structure.pk, "0.0", future.pk))
            self.to_context(**{'calc_defect_q0': future})

        # Defect structure; target charge state
        if self.inputs.run_pw_defect_q:
            overrides['base']['pw']['parameters'] = recursive_merge(overrides['base']['pw']['parameters'], {'SYSTEM':{'tot_charge': self.inputs.defect_charge.value}})
            overrides['base_final_scf']['pw']['parameters'] = recursive_merge(overrides['base_final_scf']['pw']['parameters'], {'SYSTEM':{'tot_charge': self.inputs.defect_charge.value}})

            inputs = PwRelaxWorkChain.get_builder_from_protocol(
                    code = self.inputs.qe.dft.supercell.code,
                    structure = self.inputs.defect_structure,
                    overrides = overrides,
                    relax_type = relax_type[self.inputs.relaxation_scheme.value]
                    )

            inputs['base']['pw']['metadata'] = self.inputs.qe.dft.supercell.scheduler_options.get_dict()
            inputs['base']['pw']['settings'] = self.inputs.qe.dft.supercell.settings
            inputs['base_final_scf']['pw']['metadata'] = self.inputs.qe.dft.supercell.scheduler_options.get_dict()
            inputs['base_final_scf']['pw']['settings'] = self.inputs.qe.dft.supercell.settings

            #future = self.submit(PwRelaxWorkChain, **inputs)
            future = self.submit(inputs)
            self.report(
                'Launching PWSCF for the defect structure (PK={}) with charge {} (PK={})'
                .format(self.inputs.defect_structure.pk, self.inputs.defect_charge.value, future.pk))
            self.to_context(**{'calc_defect_q': future})

    def check_dft_calcs_gaussian_correction(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """

        # Host
        if self.inputs.run_pw_host:
            host_calc = self.ctx['calc_host']
            if host_calc.is_finished_ok:
                self.ctx.host_energy = orm.Float(host_calc.outputs.output_parameters.get_dict()['energy']) # eV
                self.report('The energy of the host is: {} eV'.format(self.ctx.host_energy.value))
                #self.ctx.host_vbm = orm.Float(host_calc.outputs.output_band.get_array('bands')[0][-1]) # valence band maximum
                self.ctx.host_vbm = orm.Float(get_vbm(host_calc))
                self.report('The top of valence band is: {} eV'.format(self.ctx.host_vbm.value))
                is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(host_calc.outputs.output_band)
                if not is_insulator:
                    self.report('WARNING! The ground state of the host structure is metallic!')
            else:
                self.report(
                    'PWSCF for the host structure has failed with status {}'.format(host_calc.exit_status))
                return self.exit_codes.ERROR_DFT_CALCULATION_FAILED
        else:
            HostNode = orm.load_node(self.inputs.host_node.value)
            self.ctx.host_energy = orm.Float(HostNode.outputs.output_parameters.get_dict()['energy']) # eV
            self.report('Extracting PWSCF for host structure with charge {} from node PK={}'
                    .format("0.0", self.inputs.host_node.value))
            self.report('The energy of the host is: {} eV'.format(self.ctx.host_energy.value))
            #self.ctx.host_vbm = orm.Float(HostNode.outputs.output_band.get_array('bands')[0][-1]) # eV
            self.ctx.host_vbm = orm.Float(get_vbm(HostNode))
            self.report('The top of valence band is: {} eV'.format(self.ctx.host_vbm.value))
            is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(HostNode.outputs.output_band)
            if not is_insulator:
                self.report('WARNING! The ground state of the host structure is metallic!')

        # Defect (q=0)
        if self.inputs.run_pw_defect_q0:
            defect_q0_calc = self.ctx['calc_defect_q0']
            if not defect_q0_calc.is_finished_ok:
                self.report('PWSCF for the defect structure (with charge 0) has failed with status {}'.format(defect_q0_calc.exit_status))
                return self.exit_codes.ERROR_DFT_CALCULATION_FAILED
            else:
                self.report('The energy of neutral defect structure is: {} eV'.format(defect_q0_calc.outputs.output_parameters.get_dict()['energy']))
                is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(defect_q0_calc.outputs.output_band)
                if not is_insulator:
                    self.report('WARNING! The ground state of neutral defect structure is metallic!')
        else:
            Defect_q0Node = orm.load_node(self.inputs.defect_q0_node.value)
            self.report('Extracting PWSCF for defect structure with charge {} from node PK={}'.format("0.0", self.inputs.defect_q0_node.value))
            self.report('The energy of neutral defect structure is: {} eV'.format(Defect_q0Node.outputs.output_parameters.get_dict()['energy']))
            is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(Defect_q0Node.outputs.output_band)
            if not is_insulator:
                self.report('WARNING! The ground state of neutral defect structure is metallic!')

        # Defect (q=q)
        if self.inputs.run_pw_defect_q:
            defect_q_calc = self.ctx['calc_defect_q']
            if defect_q_calc.is_finished_ok:
                self.ctx.defect_energy = orm.Float(defect_q_calc.outputs.output_parameters.get_dict()['energy']) # eV
                self.report('The energy of defect structure with charge {} is: {} eV'.
                        format(self.inputs.defect_charge.value, defect_q_calc.outputs.output_parameters.get_dict()['energy']))
                is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(defect_q_calc.outputs.output_band)
                if not is_insulator:
                    self.report('WARNING! The ground state of charged defect structure is metallic!')
            else:
                self.report(
                    'PWSCF for the defect structure (with charge {}) has failed with status {}'
                    .format(self.inputs.defect_charge.value, defect_q_calc.exit_status))
                return self.exit_codes.ERROR_DFT_CALCULATION_FAILED
        else:
            Defect_qNode = orm.load_node(self.inputs.defect_q_node.value)
            self.report('Extracting PWSCF for defect structure with charge {} from node PK={}'
                    .format(self.inputs.defect_charge.value, self.inputs.defect_q_node.value))
            self.ctx.defect_energy = orm.Float(Defect_qNode.outputs.output_parameters.get_dict()['energy']) # eV
            self.report('The energy of defect structure with charge {} is: {} eV'.
                    format(self.inputs.defect_charge.value, Defect_qNode.outputs.output_parameters.get_dict()['energy']))
            is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(Defect_qNode.outputs.output_band)
            if not is_insulator:
                self.report('WARNING! The ground state of charged defect structure is metallic!')

    def get_dft_potentials_gaussian_correction(self):
        """
        Obtain the electrostatic potentials from the PWSCF calculations.
        """

        # User inputs
        pp_inputs = PpCalculation.get_builder()
        pp_inputs.code = self.inputs.qe.pp.code
        pp_inputs.metadata = self.inputs.qe.pp.scheduler_options.get_dict()

        # Fixed settings
        #pp_inputs.plot_number = orm.Int(0)  # Charge density
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
#        if self.inputs.run_pw_host:
#            pp_inputs.parent_folder = self.ctx['calc_host'].outputs.remote_folder
#        else:
#            HostNode = orm.load_node(int(self.inputs.host_node))
#            pp_inputs.parent_folder =  HostNode.outputs.remote_folder
        if self.inputs.run_v_host:
            if self.inputs.run_pw_host:
                pp_inputs.parent_folder = self.ctx['calc_host'].outputs.remote_folder
            else:
                temp_node = orm.load_node(self.inputs.host_node.value)
                pp_inputs.parent_folder = temp_node.outputs.remote_folder
            future = self.submit(PpCalculation, **pp_inputs)
            self.report('Launching PP.x for host structure (PK={}) with charge {} (PK={})'.
                format(self.inputs.host_structure.pk, "0.0", future.pk))
            self.to_context(**{'calc_v_host': future})
        else:
            self.ctx['calc_v_host'] = orm.load_node(self.inputs.v_host_node.value)

        # Defect (q=0)
        if self.inputs.run_v_defect_q0:
            if self.inputs.run_pw_defect_q0:
                pp_inputs.parent_folder = self.ctx['calc_defect_q0'].outputs.remote_folder
            else:
                temp_node = orm.load_node(self.inputs.defect_q0_node.value)
                pp_inputs.parent_folder = temp_node.outputs.remote_folder
            future = self.submit(PpCalculation, **pp_inputs)
            self.report('Launching PP.x for defect structure (PK={}) with charge {} (PK={})'
                .format(self.inputs.defect_structure.pk, "0.0", future.pk))
            self.to_context(**{'calc_v_defect_q0': future})
        else:
            self.ctx['calc_v_defect_q0'] = orm.load_node(self.inputs.v_defect_q0_node.value)


        # Defect (q=q)
        if self.inputs.run_v_defect_q:
            if self.inputs.run_pw_defect_q:
                pp_inputs.parent_folder = self.ctx['calc_defect_q'].outputs.remote_folder
            else:
                temp_node = orm.load_node(self.inputs.defect_q_node.value)
                pp_inputs.parent_folder = temp_node.outputs.remote_folder
            future = self.submit(PpCalculation, **pp_inputs)
            self.report('Launching PP.x for defect structure (PK={}) with charge {} (PK={})'
                .format(self.inputs.defect_structure.pk, self.inputs.defect_charge.value, future.pk))
            self.to_context(**{'calc_v_defect_q': future})
        else:
            self.ctx['calc_v_defect_q'] = orm.load_node(self.inputs.v_defect_q_node.value)

    def check_dft_potentials_gaussian_correction(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """

        # Host
        host_pp = self.ctx['calc_v_host']
        if host_pp.is_finished_ok:
            # data_array = host_pp.outputs.output_data.get_array('data')
            # v_data = orm.ArrayData()
            # v_data.set_array('data', data_array)
            # self.ctx.v_host = v_data
            # self.ctx.v_host = host_pp.outputs.output_data
            self.ctx.v_host = get_data_array(host_pp.outputs.output_data)
        else:
            self.report(
                'Post processing for the host structure has failed with status {}'.format(host_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED

        # Defect (q=0)
        defect_q0_pp = self.ctx['calc_v_defect_q0']
        if defect_q0_pp.is_finished_ok:
            # data_array = defect_q0_pp.outputs.output_data.get_array('data')
            # v_data = orm.ArrayData()
            # v_data.set_array('data', data_array)
            # self.ctx.v_defect_q0 = v_data
            # self.ctx.v_defect_q0 = defect_q0_pp.outputs.output_data
            self.ctx.v_defect_q0 = get_data_array(defect_q0_pp.outputs.output_data)
        else:
            self.report(
                'Post processing for the defect structure (with charge 0) has failed with status {}'
                .format(defect_q0_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED

        # Defect (q=q)
        defect_q_pp = self.ctx['calc_v_defect_q']
        if defect_q_pp.is_finished_ok:
            # data_array = defect_q_pp.outputs.output_data.get_array('data')
            # v_data = orm.ArrayData()
            # v_data.set_array('data', data_array)
            # self.ctx.v_defect_q = v_data
            # self.ctx.v_defect_q = defect_q_pp.outputs.output_data
            self.ctx.v_defect_q = get_data_array(defect_q_pp.outputs.output_data)
        else:
            self.report(
                'Post processing for the defect structure (with charge {}) has failed with status {}'
                .format(self.inputs.defect_charge.value,defect_q_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED

    def get_charge_density(self):
        """
        Obtain the electrostatic potentials from the PWSCF calculations.
        """

        # User inputs
        pp_inputs = PpCalculation.get_builder()
        pp_inputs.code = self.inputs.qe.pp.code
        pp_inputs.metadata = self.inputs.qe.pp.scheduler_options.get_dict()

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
        if self.inputs.run_rho_host:
            if self.inputs.run_pw_host:
                pp_inputs.parent_folder = self.ctx['calc_host'].outputs.remote_folder
            else:
                temp_node = orm.load_node(self.inputs.host_node.value)
                pp_inputs.parent_folder = temp_node.outputs.remote_folder
            future = self.submit(PpCalculation, **pp_inputs)
            self.report('Launching PP.x for charge density of host structure (PK={}) with charge {} (PK={})'
                .format(self.inputs.host_structure.pk, "0.0", future.pk))
            self.to_context(**{'calc_rho_host': future})
        else:
            self.ctx['calc_rho_host'] = orm.load_node(self.inputs.rho_host_node.value)

        # Defect (q=0)
        if self.inputs.run_rho_defect_q0:
            if self.inputs.run_pw_defect_q0:
                pp_inputs.parent_folder = self.ctx['calc_defect_q0'].outputs.remote_folder
            else:
                temp_node = orm.load_node(self.inputs.defect_q0_node.value)
                pp_inputs.parent_folder = temp_node.outputs.remote_folder
            future = self.submit(PpCalculation, **pp_inputs)
            self.report('Launching PP.x for charge density of defect structure (PK={}) with charge {} (PK={})'
                .format(self.inputs.defect_structure.pk, "0.0", future.pk))
            self.to_context(**{'calc_rho_defect_q0': future})
        else:
            self.ctx['calc_rho_defect_q0'] = orm.load_node(self.inputs.rho_defect_q0_node.value)

        # Defect (q=q)
        if self.inputs.run_rho_defect_q:
            if self.inputs.run_pw_defect_q:
                pp_inputs.parent_folder = self.ctx['calc_defect_q'].outputs.remote_folder
            else:
                temp_node = orm.load_node(self.inputs.defect_q_node.value)
                pp_inputs.parent_folder = temp_node.outputs.remote_folder
            future = self.submit(PpCalculation, **pp_inputs)
            self.report('Launching PP.x for charge density of defect structure (PK={}) with charge {} (PK={})'
                .format(self.inputs.defect_structure.pk, self.inputs.defect_charge.value, future.pk))
            self.to_context(**{'calc_rho_defect_q': future})
        else:
            self.ctx['calc_rho_defect_q'] = orm.load_node(self.inputs.rho_defect_q_node.value)

    def check_charge_density_calculations(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """

        # Host
        host_pp = self.ctx['calc_rho_host']
        if host_pp.is_finished_ok:
            # data_array = host_pp.outputs.output_data.get_array('data')
            # v_data = orm.ArrayData()
            # v_data.set_array('data', data_array)
            # self.ctx.rho_host = v_data
            # self.ctx.rho_host = host_pp.outputs.output_data
            self.ctx.rho_host = get_data_array(host_pp.outputs.output_data)
        else:
            self.report(
                'Post processing for the host structure has failed with status {}'.format(host_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED

        # Defect (q=0)
        defect_q0_pp = self.ctx['calc_rho_defect_q0']
        if defect_q0_pp.is_finished_ok:
            # data_array = defect_q0_pp.outputs.output_data.get_array('data')
            # v_data = orm.ArrayData()
            # v_data.set_array('data', data_array)
            # self.ctx.rho_defect_q0 = v_data
            # self.ctx.rho_defect_q0 = defect_q0_pp.outputs.output_data
            self.ctx.rho_defect_q0 = get_data_array(defect_q0_pp.outputs.output_data)
        else:
            self.report(
                'Post processing for the defect structure (with charge 0) has failed with status {}'
                .format(defect_q0_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED

        # Defect (q=q)
        defect_q_pp = self.ctx['calc_rho_defect_q']
        if defect_q_pp.is_finished_ok:
            # data_array = defect_q_pp.outputs.output_data.get_array('data')
            # v_data = orm.ArrayData()
            # v_data.set_array('data', data_array)
            # self.ctx.rho_defect_q = v_data
            # self.ctx.rho_defect_q = defect_q_pp.outputs.output_data
            self.ctx.rho_defect_q = get_data_array(defect_q_pp.outputs.output_data)
        else:
            self.report(
                'Post processing for the defect structure (with charge 0) has failed with status {}'
                .format(defect_q_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED

    def prep_hostcell_calc_for_dfpt(self):
        """
        Run a DFT calculation on the structure to be used for the computation of the
        dielectric constant
        """
        self.report("An alternative unit cell has been requested")

        # Another code may be desirable - N.B. in AiiDA a code refers to a specific
        # executable on a specific computer. As the PH calculation may have to be run on
        # an HPC cluster, the PW calculation must be run on the same machine and so this
        # may necessitate that a different code is used than that for the supercell calculations.

        relax_type = {'fixed': RelaxType.NONE, 'relax': RelaxType.POSITIONS, 'vc-relax': RelaxType.POSITIONS_CELL}

        overrides = {
                'base':{
                    # 'pseudo_family': self.inputs.qe.dft.unitcell.pseudopotential_family.value,
                    'pw': {
                    	'parameters': {},
                        # 'metadata': self.inputs.qe.dft.unitcell.scheduler_options.get_dict(),
                        'settings': self.inputs.qe.dft.unitcell.settings.get_dict(),
                        }
                    },
                'base_final_scf':{
                    # 'pseudo_family': self.inputs.qe.dft.unitcell.pseudopotential_family.value,
                    'pw': {
                    	'parameters': {},
                        # 'metadata': self.inputs.qe.dft.unitcell.scheduler_options.get_dict(),
                        'settings': self.inputs.qe.dft.unitcell.settings.get_dict(),
                        }
                    },
                'clean_workdir' : orm.Bool(False),
                }

        if 'pseudopotential_family' in self.inputs.qe.dft.unitcell:
        	overrides['base']['pseudo_family'] = self.inputs.qe.dft.unitcell.pseudopotential_family.value
        	overrides['base_final_scf']['pseudo_family'] = self.inputs.qe.dft.unitcell.pseudopotential_family.value
        if 'parameters' in self.inputs.qe.dft.unitcell:
            overrides['base']['pw']['parameters'] = self.inputs.qe.dft.unitcell.parameters.get_dict()
            overrides['base_final_scf']['pw']['parameters'] = self.inputs.qe.dft.unitcell.parameters.get_dict()

        inputs = PwRelaxWorkChain.get_builder_from_protocol(
                    code = self.inputs.qe.dft.unitcell.code,
                    structure = self.inputs.host_unitcell,
                    overrides = overrides,
                    relax_type = relax_type[self.inputs.relaxation_scheme.value]
                    )

        inputs['base']['pw']['metadata'] = self.inputs.qe.dft.unitcell.scheduler_options.get_dict()
        inputs['base']['pw']['settings'] = self.inputs.qe.dft.unitcell.settings
        inputs['base_final_scf']['pw']['metadata'] = self.inputs.qe.dft.unitcell.scheduler_options.get_dict()
        inputs['base_final_scf']['pw']['settings'] = self.inputs.qe.dft.unitcell.settings

        #future = self.submit(PwRelaxWorkChain, **inputs)
        future = self.submit(inputs)
        self.report(
            'Launching PWSCF for host unitcell structure (PK={}) at node (PK={})'.
            format(self.inputs.host_unitcell.pk, future.pk))
        self.to_context(**{'calc_host_unitcell': future})

    def check_hostcell_calc_for_dfpt(self):
        """
        Check if the DFT calculation to be used for the computation of the
        dielectric constant has completed successfully.
        """

        host_unitcell_calc = self.ctx['calc_host_unitcell']
        if not host_unitcell_calc.is_finished_ok:
            self.report(
                'PWSCF for the host unitcell structure has failed with status {}'.
                format(host_unitcell_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED

    def prep_calc_dfpt_calculation(self):
        """
        Run a DFPT calculation to compute the dielectric constant for the pristine material
        """

        ph_inputs = self.inputs.qe.dfpt.code.get_builder()

        # Setting up the calculation depends on whether the parent SCF calculation is either
        # the host supercell or an alternative host unitcell
        if self.inputs.host_unitcell:
            ph_inputs.parent_folder = self.ctx['calc_host_unitcell'].outputs.remote_folder
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
            qpoints.set_cell_from_structure(structuredata=self.ctx['calc_host_unitcell'].inputs.structure)
        else:
            qpoints.set_cell_from_structure(structuredata=self.ctx['calc_host'].inputs.structure)
        qpoints.set_kpoints([[0.,0.,0.]])
        qpoints.get_kpoints(cartesian=True)
        ph_inputs.qpoints = qpoints

        ph_inputs.metadata = self.inputs.qe.dfpt.scheduler_options.get_dict()

        future = self.submit(ph_inputs)
        self.report('Launching PH for host structure (PK={})'.format(self.inputs.host_structure.pk, future.pk))
        self.to_context(**{'calc_dfpt': future})

    def get_permittivity(self):
        """
        Compute the dielectric constant to be used in the correction
        """
        if self.inputs.run_dfpt:
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

