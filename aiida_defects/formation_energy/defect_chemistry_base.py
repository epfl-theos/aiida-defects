# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida import orm
from aiida.engine import WorkChain, calcfunction, ToContext, if_, submit
import numpy as np
from .chemical_potential.chemical_potential import ChemicalPotentialWorkchain

# from .utils import (
#     get_raw_formation_energy,
#     get_corrected_formation_energy,
#     get_corrected_aligned_formation_energy,
# )
from .utils import *

class DefectChemistryWorkchainBase(WorkChain):
    """
    The base class to determine the defect chemistry of a given material, containing the
    generic, code-agnostic methods, error codes, etc. Defect chemistry refer to the concentration or defect formation
    energy of all possible defects (vacancies, interstitials, substitutions,...) which can exist in the material at
    thermodynamics equilibrium.

    Any computational code can be used to calculate the required energies and relative permittivity.
    However, different codes must be setup in specific ways, and so separate classes are used to implement these
    possibilities. This is an abstract class and should not be used directly, but rather the
    concrete code-specific classes should be used instead.
    """

    @classmethod
    def define(cls, spec):
        super(DefectChemistryWorkchainBase, cls).define(spec)

        spec.input('restart_wc', valid_type=orm.Bool, required=False, default=lambda: orm.Bool(False),
            help="whether to restart the workchain from previous run or to start from scratch")
        spec.input('restart_node', valid_type=orm.Int, required=False,
            help="if restart from previous run, provide the node corresponding to that run")
        spec.input("unitcell", valid_type=orm.StructureData,
            help="Pristine structure to use in the calculation of permittivity and DOS", required=False)
        spec.input("host_structure", valid_type=orm.StructureData,
           help="Pristine supercell without defect")
        spec.input('defect_info', valid_type=orm.Dict,
            help="Dictionary containing the information about defects included in the calculations of defect chemistry")

        # Chemical potential workchain
        spec.expose_inputs(ChemicalPotentialWorkchain)

        # Fermi level workchain
        spec.input("temperature", valid_type=orm.Float,
            help="temperature at which the defect chemistry is determined. Enter in the calculaion of electron, hole and defect concentrations")
        spec.input("dopant", valid_type=orm.Float, required=False,
            help="Charge and concentration of aliovalent dopants added to the system. Used in the 'frozen' defect approach")

        # Input Correction workchain
        spec.input("correction_scheme", valid_type=orm.Str,
            help="The correction scheme to apply")
        # Charge Model Settings
        spec.input_namespace('charge_model',
            help="Namespace for settings related to different charge models")
        spec.input("charge_model.model_type",
            valid_type=orm.Str,
            help="Charge model type: 'fixed' or 'fitted'",
            default=lambda: orm.Str('fixed'))
        # Fixed
        spec.input_namespace('charge_model.fixed', required=False, populate_defaults=False,
            help="Inputs for a fixed charge model using a user-specified multivariate gaussian")
        spec.input("charge_model.fixed.covariance_matrix",
            valid_type=orm.ArrayData,
            help="The covariance matrix used to construct the gaussian charge distribution.")
        # Fitted
        spec.input_namespace('charge_model.fitted', required=False, populate_defaults=False,
            help="Inputs for a fitted charge model using a multivariate anisotropic gaussian.")
        spec.input("charge_model.fitted.tolerance",
            valid_type=orm.Float,
            help="Permissable error for any fitted charge model parameter.",
            default=lambda: orm.Float(1.0e-3))
        spec.input("charge_model.fitted.strict_fit",
            valid_type=orm.Bool,
            help="When true, exit the workchain if a fitting parameter is outside the specified tolerance.",
            default=lambda: orm.Bool(True))
        spec.input("epsilon", valid_type=orm.ArrayData, required=False,
            help="Dielectric constant of the host")
        spec.input("cutoff",
            valid_type=orm.Float,
            default=lambda: orm.Float(100.),
            help="Plane wave cutoff for electrostatic model.")

        # Outputs
        spec.output(
            "defect_formation_energy", valid_type=orm.Dict, required=True)
        spec.output(
            "chemical_potential", valid_type=orm.Dict, required=True)
        spec.output(
            "fermi_level", valid_type=orm.Dict, required=True)
#        spec.output(
#            "defect_data", valid_type=orm.Dict, required=True)
        spec.output(
            "total_correction", valid_type=orm.Dict, required=True)
        spec.output(
            "electrostatic_correction", valid_type=orm.Dict, required=True)
        spec.output(
            "potential_alignment", valid_type=orm.Dict, required=True)

        # Error codes
        spec.exit_code(201, "ERROR_INVALID_CORRECTION",
            message="The requested correction scheme is not recognised")
        spec.exit_code(202, "ERROR_PARAMETER_OVERRIDE",
            message="Input parameter dictionary key cannot be set explicitly")
        spec.exit_code(301, "ERROR_CORRECTION_WORKCHAIN_FAILED",
            message="The correction scheme sub-workchain failed")
        spec.exit_code(302, "ERROR_DFT_CALCULATION_FAILED",
            message="DFT calculation failed")
        spec.exit_code(303, "ERROR_PP_CALCULATION_FAILED",
            message="A post-processing calculation failed")
        spec.exit_code(304, "ERROR_DFPT_CALCULATION_FAILED",
            message="DFPT calculation failed")
        spec.exit_code(305, "ERROR_DOS_CALCULATION_FAILED",
            message="DOS calculation failed")
        spec.exit_code(306, "ERROR_DOS_INTEGRATION_FAILED",
            message="The number of electrons obtained from the integration of the DOS is different from the expected number of electrons")
        spec.exit_code(401, "ERROR_CHEMICAL_POTENTIAL_WORKCHAIN_FAILED",
            message="The chemical potential calculation failed")
        spec.exit_code(402, "ERROR_FERMI_LEVEL_WORKCHAIN_FAILED",
            message="The self-consistent fermi level calculation failed")
        spec.exit_code(500, "ERROR_PARAMETER_OVERRIDE",
            message="Input parameter dictionary key cannot be set explicitly")
        spec.exit_code(999, "ERROR_NOT_IMPLEMENTED",
            message="The requested method is not yet implemented")

    def setup(self):
        """
        Setup the workchain
        """

        self.ctx.inputs_chempots = self.exposed_inputs(ChemicalPotentialWorkchain)

        # Check if correction scheme is valid:
        correction_schemes_available = ["gaussian", "point"]
        if self.inputs.correction_scheme is not None:
            if self.inputs.correction_scheme not in correction_schemes_available:
                return self.exit_codes.ERROR_INVALID_CORRECTION

        self.ctx.all_dopants = self.inputs.formation_energy_dict.get_dict()
        self.ctx.chempot_dopants = self.ctx.all_dopants
        self.ctx.sc_fermi_dopants = list(self.inputs.formation_energy_dict.get_dict().keys())
        #self.ctx.pw_host_dopants = list(self.inputs.formation_energy_dict.get_dict().keys())
        self.ctx.pw_host_dopants = ['intrinsic']


        self.ctx['output_unitcell'] = {}
        #self.ctx['calc_dos'] = {}
        self.ctx.dos = {}
        self.ctx.all_defects = self.inputs.defect_info.get_dict()

        self.ctx.defect_data = {}
        self.ctx.chemical_potential = {}
        self.ctx.fermi_level = {}
        self.ctx.defect_formation_energy = {}

        self.ctx.pw_defects = self.inputs.defect_info.get_dict()
        self.ctx.phi_defects = self.inputs.defect_info.get_dict()
        self.ctx.rho_defects = self.inputs.defect_info.get_dict()
        self.ctx.gc_correction_defects = self.inputs.defect_info.get_dict()

        self.ctx.total_correction = {}
        self.ctx.electrostatic_correction = {}
        self.ctx.potential_alignment = {}

        # defect_data contains all the information requires to compute defect formation energy such as E_corr, E_host, vbm,...

        for defect, properties in self.ctx.all_defects.items():
            self.ctx.total_correction[defect] = {}
            self.ctx.electrostatic_correction[defect] = {}
            self.ctx.potential_alignment[defect] = {}
            # self.ctx.defect_data[defect] = {'N_site': properties['N_site'], 'species': properties['species'], 'charges': {}}
            # Add neutral defect to the calculation even if the user doesn't specify it because it is needed to generate the charge model
            if 0.0 not in properties['charges']:
                self.ctx.all_defects[defect]['charges'].append(0.0)
                self.ctx.pw_defects[defect]['charges'].append(0.0)
                self.ctx.phi_defects[defect]['charges'].append(0.0)
                self.ctx.rho_defects[defect]['charges'].append(0.0)
            # for chg in self.ctx.all_defects[defect]['charges']:
            #     self.ctx.defect_data[defect]['charges'][str(chg)] = {}
        #self.report('The defect data are: {}'.format(self.ctx.defect_data))


    def if_restart_wc(self):
        return self.inputs.restart_wc.value

    def if_rerun_calc_unitcell(self):
        if not self.ctx['output_unitcell']:
            return True
        else:
            self.ctx.number_of_electrons = self.ctx.output_unitcell['number_of_electrons']
            self.ctx.vbm = self.ctx.output_unitcell['vbm']
            self.ctx.band_gap = self.ctx.output_unitcell['band_gap']
            return False

    def if_rerun_calc_dos(self):
        if not self.ctx.dos:
            #self.report('start from scratch')
            return True
        else:
            #sefl.out('density_of_states', store_dos(self.ctx.dos))
            return False

    def if_run_dfpt(self):
        return self.inputs.epsilon == 0.0

    def run_chemical_potential_workchain(self):
        from .chemical_potential.chemical_potential import (
                ChemicalPotentialWorkchain, )

        self.report('Computing the chemical potentials...')
        inputs = {
            "compound": self.ctx.inputs_chempots.compound,
            "dependent_element": self.ctx.inputs_chempots.dependent_element,
            "ref_energy": self.ctx.inputs_chempots.ref_energy,
            "tolerance": self.ctx.inputs_chempots.tolerance,
        }

        for key, ef_dict in self.ctx.chempot_dopants.items():
            inputs['formation_energy_dict'] = orm.Dict(dict=ef_dict)
            if key != 'intrinsic':
                inputs['dopant_elements'] = orm.List(list=[key])

            workchain_future = self.submit(ChemicalPotentialWorkchain, **inputs)
            workchain_future.label = key
            label = "chem_potential_wc_{}".format(key)
            self.to_context(**{label: workchain_future})
            # Re-initialize dopant_elements to []
            inputs['dopant_elements'] = orm.List(list=[])

    def check_chemical_potential_workchain(self):
        """
        Check if the chemical potential workchain have finished correctly.
        If yes, assign the output to context
        """

        # self.ctx["chem_potential_wc_N"] = orm.load_node(230917)
        # self.ctx["chem_potential_wc_intrinsic"] = orm.load_node(230921)

        for key, ef_dict in self.ctx.all_dopants.items():
            chem_potential_wc = self.ctx["chem_potential_wc_{}".format(key)]
            if not chem_potential_wc.is_finished_ok:
                self.report(
                    "Chemical potential workchain failed with status {}".format(
                        chem_potential_wc.exit_status
                    )
                )
                return self.exit_codes.ERROR_CHEMICAL_POTENTIAL_WORKCHAIN_FAILED
            else:
                self.ctx.chemical_potential[key] = chem_potential_wc.outputs.chemical_potential
            # self.report('Chemical potentials: {}'.format(self.ctx.chemical_potential[key].get_dict()))
        self.out('chemical_potential', store_dict(**self.ctx.chemical_potential))

    def run_gaussian_correction_workchain(self):
        """
        Run the workchain for the Gaussian Countercharge correction
        """
        from .corrections.gaussian_countercharge.gaussian_countercharge import (
            GaussianCounterChargeWorkchain,
        )

        self.report("Computing correction via the Gaussian Countercharge scheme")

        # parent_node = orm.load_node(224010)
        # self.ctx.phi_host = get_data_array(parent_node.inputs.v_host)
        # self.ctx.rho_host = get_data_array(parent_node.inputs.rho_host)
        # self.ctx['phi_defect_N-O[0.0]'] = get_data_array(parent_node.inputs.v_defect_q0)
        # self.ctx['phi_defect_N-O[-1.0]'] = get_data_array(parent_node.inputs.v_defect_q)
        # self.ctx['rho_defect_N-O[-1.0]'] = get_data_array(parent_node.inputs.rho_defect_q)

        # parent_node = orm.load_node(224014)
        # self.ctx['phi_defect_V_Cl[0.0]'] = get_data_array(parent_node.inputs.v_defect_q0)
        # self.ctx['phi_defect_V_Cl[1.0]'] = get_data_array(parent_node.inputs.v_defect_q)
        # self.ctx['rho_defect_V_Cl[1.0]'] = get_data_array(parent_node.inputs.rho_defect_q)

        inputs = {
            "v_host": self.ctx.phi_host,
            "rho_host": self.ctx.rho_host,
            "host_structure": self.inputs.host_structure,
            "epsilon": self.ctx.epsilon,
            "cutoff" : self.inputs.cutoff,
            'charge_model': {
                'model_type': self.inputs.charge_model.model_type
                }
        }

        #defect_info = self.ctx.all_defects
        for defect, properties in self.ctx.gc_correction_defects.items():
            print(defect, properties)
            inputs['defect_site'] = orm.List(list=properties['defect_position'])
            inputs["v_defect_q0"] = self.ctx['phi_defect_{}[{}]'.format(defect, 0.0)]
            for chg in properties['charges']:
                if chg != 0.0:
                    inputs["defect_charge"] = orm.Float(chg)
                    inputs["v_defect_q"] = self.ctx['phi_defect_{}[{}]'.format(defect, chg)]
                    inputs["rho_defect_q"] = self.ctx['rho_defect_{}[{}]'.format(defect, chg)]

                    if self.inputs.charge_model.model_type.value == 'fixed':
                        inputs['charge_model']['fixed'] = {'covariance_matrix': self.inputs.charge_model.fixed.covariance_matrix}
                    else:
                        inputs['charge_model']['fitted'] = {'tolerance': self.inputs.charge_model.fitted.tolerance,
                                                            'strict_fit': self.inputs.charge_model.fitted.strict_fit}

                    workchain_future = self.submit(GaussianCounterChargeWorkchain, **inputs)
                    label = "correction_wc_{}[{}]".format(defect, chg)
                    workchain_future.label = label
                    self.to_context(**{label: workchain_future})

    def check_correction_workchain(self):
        """
        Check if the potential alignment workchains have finished correctly.
        If yes, assign the outputs to the context
        """

        # self.ctx["correction_wc_N-O[-1.0]"] = orm.load_node(231218)
        # self.ctx["correction_wc_V_Cl[1.0]"] = orm.load_node(231222)

        total_correction = {}
        electrostatic_correction = {}
        potential_alignment = {}
        for defect, properties in self.ctx.all_defects.items():
            temp_total = {}
            temp_electrostatic = {}
            temp_alignment ={}
            for chg in properties['charges']:
                # print(defect, chg)
                if chg != 0.0:
                    correction_wc = self.ctx["correction_wc_{}[{}]".format(defect, chg)]
                    if not correction_wc.is_finished_ok:
                        self.report("Correction workchain failed with status {}"
                                .format(correction_wc.exit_status)
                                )
                        return self.exit_codes.ERROR_CORRECTION_WORKCHAIN_FAILED
                    else:
                        temp_total[convert_key(str(chg))] = correction_wc.outputs.total_correction
                        temp_electrostatic[convert_key(str(chg))] = correction_wc.outputs.electrostatic_correction
                        temp_alignment[convert_key(str(chg))] = correction_wc.outputs.potential_alignment
                        # self.ctx.defect_data[defect]['charges'][str(chg)]['E_corr'] = correction_wc.outputs.total_correction.value
                else:
                    temp_total[convert_key('0.0')] = orm.Float(0.0)
                    temp_electrostatic[convert_key('0.0')] = orm.Float(0.0)
                    temp_alignment[convert_key('0.0')] = orm.Float(0.0)
                    # self.ctx.defect_data[defect]['charges']['0.0']['E_corr'] = 0.0
            total_correction[convert_key(defect)] = store_dict(**temp_total)
            electrostatic_correction[convert_key(defect)] = store_dict(**temp_electrostatic)
            potential_alignment[convert_key(defect)] = store_dict(**temp_alignment)
        self.ctx.total_correction = store_dict(**total_correction)
        self.ctx.electrostatic_correction = store_dict(**electrostatic_correction)
        self.ctx.potential_alignment = store_dict(**potential_alignment)
        # self.out('defect_data', store_dict(orm.Dict(dict=self.ctx.defect_data)))
        self.out('total_correction', self.ctx.total_correction)
        self.out('electrostatic_correction', self.ctx.electrostatic_correction)
        self.out('potential_alignment', self.ctx.potential_alignment)
        # self.report('The defect data are: {}'.format(self.ctx.defect_data))

    def create_defect_data(self):

        compound = self.inputs.compound.value
        for dopant in self.ctx.sc_fermi_dopants:
            pw_calc_outputs = {}
            # for defect, properties in self.ctx.all_defect.items():
            for defect, properties in self.inputs.defect_info.get_dict().items():
                if is_intrinsic_defect(properties['species'], compound) or dopant in properties['species'].keys():
                    for chg in properties['charges']:
                        pw_calc_outputs[convert_key(defect)+'_'+convert_key(str(chg))] = self.ctx['calc_defect_{}[{}]'.format(defect, chg)].outputs.output_parameters
            self.ctx.defect_data[dopant] = get_defect_data(orm.Str(dopant),
                                                            self.inputs.compound,
                                                            self.inputs.defect_info,
                                                            self.ctx.vbm,
                                                            self.ctx['calc_host_intrinsic'].outputs.output_parameters,
                                                            self.ctx.total_correction,
                                                            **pw_calc_outputs)
            self.report('Defect data {}: {}'.format(dopant, self.ctx.defect_data[dopant].get_dict()))

    def run_fermi_level_workchain(self):
        from .fermi_level.fermi_level import (
                FermiLevelWorkchain, )

        self.report('Running the fermi level workchain...')

        # #self.ctx.defect_data = orm.load_node(224094).get_dict()
        # self.ctx.vbm = orm.load_node(224104).value
        # self.ctx.number_of_electrons = orm.load_node(224105).value
        # self.ctx.band_gap = orm.load_node(224106).value
        # self.ctx.dos = orm.load_node(223757)

        inputs = {
            "temperature": self.inputs.temperature,
            "valence_band_maximum": self.ctx.vbm,
            "number_of_electrons": orm.Float(self.ctx.number_of_electrons),
            "unitcell": self.inputs.unitcell,
            "DOS": self.ctx.dos,
            "band_gap": orm.Float(self.ctx.band_gap),
            #"dopant": Dict(dict={'X_1':{'c': 1E18, 'q':-1}})
        }

        for dopant in self.ctx.sc_fermi_dopants:
            inputs['defect_data'] = self.ctx.defect_data[dopant]
            # self.report('Defect data {}: {}'.format(dopant, defect_temp))
            inputs['chem_potentials'] = self.ctx["chem_potential_wc_{}".format(dopant)].outputs.chemical_potential
            workchain_future = self.submit(FermiLevelWorkchain, **inputs)
            label = "fermi_level_wc_{}".format(dopant)
            workchain_future.label = dopant
            self.to_context(**{label: workchain_future})

    def check_fermi_level_workchain(self):
        """
        Check if the fermi level workchain have finished correctly.
        If yes, assign the output to context
        """

        #for dopant, ef_dict in self.ctx.all_dopants.items():
        for dopant in self.ctx.sc_fermi_dopants:
            fermi_level_wc = self.ctx["fermi_level_wc_{}".format(dopant)]
            if not fermi_level_wc.is_finished_ok:
                self.report(
                    "Fermi level workchain of {} defect failed with status {}".format(
                        dopant, fermi_level_wc.exit_status))
                return self.exit_codes.ERROR_FERMI_LEVEL_WORKCHAIN_FAILED
            else:
                self.ctx.fermi_level[dopant] = fermi_level_wc.outputs.fermi_level
                # self.ctx.fermi_level[dopant] = fermi_level_wc.outputs.fermi_level.get_array('data').item() # get the value from 0-d numpy array
            # self.report('Fermi level: {}'.format(self.ctx.fermi_level[dopant].get_array('data')))
        self.out('fermi_level', store_dict(**self.ctx.fermi_level))

    def compute_defect_formation_energy(self):
        '''
        Computing the defect formation energies of all defects considered in the materials.
        '''

        #self.report('The defect data is :{}'.format(self.ctx.defect_data))
        #self.report('All dopants: {}'.format(self.ctx.all_dopants))
        #self.report('The potential alignment is :{}'.format(self.ctx.potential_alignment))
        #self.report('The chemical potentials are :{}'.format(self.ctx.chemical_potential))
        #self.report('The fermi level are :{}'.format(self.ctx.fermi_level))

        for dopant in self.ctx.sc_fermi_dopants:
            self.ctx.defect_formation_energy[dopant] = get_defect_formation_energy(
                    self.ctx.defect_data[dopant],
                    self.ctx.fermi_level[dopant],
                    self.ctx.chemical_potential[dopant],
                    self.ctx.potential_alignment,
                    # self.inputs.compound
                    )

        # self.report('The defect formation energy is :{}'.format(self.ctx.defect_formation_energy.get_dict()))
        self.out("defect_formation_energy", store_dict(**self.ctx.defect_formation_energy))
