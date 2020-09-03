# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

import numpy as np

from aiida import orm
from aiida.engine import WorkChain, calcfunction, ToContext, if_, submit
from aiida.plugins import WorkflowFactory
from aiida_siesta.workflows.base import SiestaBaseWorkChain

from .formation_energy_base import FormationEnergyWorkchainBase
#from .utils import run_siesta_calculation
from .utils import get_raw_formation_energy, get_corrected_formation_energy, get_corrected_aligned_formation_energy

from aiida.common import AttributeDict
from aiida_siesta.calculations.tkdict import FDFDict
#from aiida_siesta.data.common import get_pseudos_from_structure
#from aiida_siesta.data.psf import PsfData
#from aiida_siesta.data.psml import PsmlData
def prepare_pseudo_inputs(structure, pseudos):
    """
    Reading Pseudos
    """
    if pseudos is None :
        raise ValueError('neither an explicit pseudos dictionary was specified')
    for kind in structure.get_kind_names():
        if kind not in pseudos:
            raise ValueError('no pseudo available for element {}'.format(kind))

    return pseudos



class FormationEnergyWorkchainSIESTA(FormationEnergyWorkchainBase):
    """
    Compute the formation energy for a given defect using SIESTA
    """
    @classmethod
    def define(cls, spec):
        super(FormationEnergyWorkchainSIESTA, cls).define(spec)

        # DFT and DFPT calculations with QuantumESPRESSO are handled with different codes, so here
        # we keep track of things with two separate namespaces. An additional code, and an additional
        # namespace, is used for postprocessing
        #spec.expose_inputs(SiestaBaseWorkChain, exclude=('metadata',))
        #spec.inputs._ports['pseudos'].dynamic = True  #Temporary fix to issue #135 plumpy
        # spec.inputs._ports["pseudos.defect"].dynamic = True
        # DFT inputs for Host (SIESTA)
        spec.input_namespace("siesta.dft.supercell_host",
            help="The siesta code to use for the calculations")
        spec.input_namespace("siesta.dft.supercell_defect_q0",
            help="The siesta code to use for the calculations")
        spec.input_namespace("siesta.dft.supercell_defect_q",
            help="The siesta code to use for the calculations")
        spec.input_namespace('pseudos_host', required=False, dynamic=True)
        spec.input_namespace('pseudos_defect', required=False, dynamic=True)
        # HOST Inputs
        spec.input("siesta.dft.supercell_host.code",
            valid_type=orm.Code,
            help="The siesta code to use for the calculations")
        #spec.input_namespace('pseudos_host'
        #    valid_type=(PsfData,PsmlData),
        #    help='Input pseudo potentials',dynamic=True)
        spec.input("siesta.dft.supercell_host.kpoints",
            valid_type=orm.KpointsData,
            help="The k-point grid to use for the calculations")
        spec.input("siesta.dft.supercell_host.basis",
            valid_type=orm.Dict,
            help="The siesta basis to use for the host calculations")
        spec.input("siesta.dft.supercell_host.parameters",
            valid_type=orm.Dict,
            help="Parameters for the SIESTA calcuations. Some will be set automatically")
        spec.input("siesta.dft.supercell_host.options",
            valid_type=orm.Dict,
            help="options for the SIESTA calcuations")
        # Defect_q0 without charge
        spec.input("siesta.dft.supercell_defect_q0.code",
            valid_type=orm.Code,
            help="The siesta code to use for the calculations")
        #spec.input_namespace('pseudos_q0',
        #    valid_type=(PsfData,PsmlData),
        #    help='Input pseudo potentials',dynamic=True)
        spec.input("siesta.dft.supercell_defect_q0.kpoints",
            valid_type=orm.KpointsData,
            help="The k-point grid to use for the calculations")
        spec.input("siesta.dft.supercell_defect_q0.basis",
            valid_type=orm.Dict,
            help="The siesta basis to use for the host calculations")
        spec.input("siesta.dft.supercell_defect_q0.parameters",
            valid_type=orm.Dict,
            help="Parameters for the SIESTA calcuations. Some will be set automatically")
        spec.input("siesta.dft.supercell_defect_q0.options",
            valid_type=orm.Dict,
            help="options for the SIESTA calcuations.")
        # DFT inputs for Defect With Charge (SIESTA)
        spec.input("siesta.dft.supercell_defect_q.code",
            valid_type=orm.Code,
            help="The siesta code to use for the calculations")
        #spec.input_namespace('pseudos_q',
        #    valid_type=(PsfData,PsmlData),
        #    help='Input pseudo potentials',dynamic=True)
        spec.input("siesta.dft.supercell_defect_q.kpoints",
            valid_type=orm.KpointsData,
            help="The k-point grid to use for the calculations")
        spec.input("siesta.dft.supercell_defect_q.basis",
            valid_type=orm.Dict,
            help="The siesta basis to use for the host calculations")
        spec.input("siesta.dft.supercell_defect_q.parameters",
            valid_type=orm.Dict,
            help="Parameters for the SIESTA calcuations. Some will be set automatically")
        spec.input("siesta.dft.supercell_defect_q.options",
            valid_type=orm.Dict,
            help="options for the SIESTA calcuations.")

        #===================================================================
        # The Steps of Workflow
        #===================================================================

        spec.outline(
                cls.setup,
                cls.run_dft_calcs,
                if_(cls.correction_required)(
                    if_(cls.is_gaussian_scheme)(
                        cls.raise_not_implemented
                        )).else_(
                        cls.check_dft_calcs,
                        cls.compute_neutral_formation_energy,
                        cls.compute_charged_formation_energy_no_corre))
                #if_(cls.is_none_scheme)(
                #    cls.check_dft_calcs,
                #    cls.compute_no_corrected_formation_energy))    
                        
                        
    
    #=========================================================================
    # This function is for running Host and defect structure neutral & charge 
    #========================================================================
    def run_dft_calcs(self):
 
        """
        Submit All DFT Calculations
        """
        self.report("Setting Up the No correction Formation Energy Workchain ")
        #--------------
        # For the Host
        #-------------
        siesta_inputs = self.inputs.siesta.dft.supercell_host.code.get_builder()
        #siesta_inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain)) For reusing 
        pseudos = None
        if "pseudos_host" in self.inputs:  #in case in the future Issue #142 will be solved
            if self.inputs.pseudos_host:
                pseudos = self.inputs.pseudos_host
        #structure = None
        structure = self.inputs.host_structure
        siesta_inputs.pseudos = prepare_pseudo_inputs(structure, pseudos)
        siesta_inputs.structure = structure
        siesta_inputs.parameters = self.inputs.siesta.dft.supercell_host.parameters
        siesta_inputs.kpoints = self.inputs.siesta.dft.supercell_host.kpoints
        siesta_inputs.metadata["options"] = self.inputs.siesta.dft.supercell_host.options.get_dict()
        siesta_inputs.basis = self.inputs.siesta.dft.supercell_host.basis #get_dict()
        #future = self.submit(SiestaBaseWorkChain,**siesta_inputs)
        future = self.submit(siesta_inputs)
        self.report(
            'Workflow Launching SIESTA for host structure (PK={})  (PK={})'
            .format(structure.pk, future.pk))
            #.format(self.inputs.host_structure.pk, future.pk))
        self.to_context(**{'calc_host': future})        
        #-------------------------------------------
        # For Defect structure; neutral charge state
        #------------------------------------------
        siesta_inputs = self.inputs.siesta.dft.supercell_defect_q0.code.get_builder()
        #siesta_inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))
        pseudos = None
        if "pseudos_defect" in self.inputs:  #in case in the future Issue #142 will be solved
            if self.inputs.pseudos_defect:
                pseudos = self.inputs.pseudos_defect
        structure = self.inputs.defect_structure
        siesta_inputs.pseudos = prepare_pseudo_inputs(structure, pseudos)
        siesta_inputs.structure = structure
        siesta_inputs.parameters = self.inputs.siesta.dft.supercell_defect_q0.parameters
        siesta_inputs.kpoints = self.inputs.siesta.dft.supercell_defect_q0.kpoints
        siesta_inputs.metadata["options"] = self.inputs.siesta.dft.supercell_defect_q0.options.get_dict()
        siesta_inputs.basis = self.inputs.siesta.dft.supercell_defect_q0.basis
        #future = self.submit(SiestaBaseWorkChain,**siesta_inputs)
        future = self.submit(siesta_inputs)
        self.report(
            'Launching SIESTA for defect structure (PK={}) with charge {} (PK={})'
            .format(structure.pk, "0.0", future.pk))
        self.to_context(**{'calc_defect_q0': future})
        #------------------------------------------
        # For Defect structure; target charge state
        #-----------------------------------------
        siesta_inputs = self.inputs.siesta.dft.supercell_defect_q.code.get_builder()
        # siesta_inputs = AttributeDict(self.exposed_inputs(SiestaBaseWorkChain))
        pseudos = None
        if "pseudos_defect" in self.inputs:  #in case in the future Issue #142 will be solved
            if self.inputs.pseudos_defect:
                pseudos = self.inputs.pseudos_defect
        structure = self.inputs.defect_structure 
        siesta_inputs.pseudos = prepare_pseudo_inputs(structure, pseudos)
        siesta_inputs.structure = structure
        siesta_inputs.parameters = self.inputs.siesta.dft.supercell_defect_q.parameters
        siesta_inputs.kpoints = self.inputs.siesta.dft.supercell_defect_q.kpoints
        siesta_inputs.metadata["options"] = self.inputs.siesta.dft.supercell_defect_q.options.get_dict()
        siesta_inputs.basis = self.inputs.siesta.dft.supercell_defect_q.basis
        #future = self.submit(SiestaBaseWorkChain,**siesta_inputs)
        future = self.submit(siesta_inputs)
        self.report(
            'Launching SIESTA for defect structure (PK={}) with charge {} (PK={})'
            .format(structure.pk,
                    self.inputs.defect_charge.value, future.pk))
        self.to_context(**{'calc_defect_q': future})
    #=========================================
    # Retrieving DFT Calculations results 
    #=========================================
    def check_dft_calcs(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """
        import sisl
        
        # We used sisl library read the Potential Grids

        self.report("Checking Up Whether DFT Caclucations are Finished ")
        # Host
        host_calc = self.ctx['calc_host']
        if host_calc.is_finished_ok:
            self.ctx.host_energy = orm.Float(host_calc.outputs.output_parameters.get_dict()['E_KS']) # eV
            self.ctx.host_vbm = orm.Float(0.0)
            self.ctx.host_VT = sisl.get_sile(host_calc.outputs.remote_folder.get_remote_path()+"/aiida.VT")    
           # self.ctx.host_vbm = orm.Float(host_calc.outputs.output_band.get_array('bands')[0][-1]) # valence band maximum
        else:
            self.report(
                'SIESTA for the host structure has failed with status {}'.
                format(host_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED

        # Defect (q=0)
        defect_q0_calc = self.ctx['calc_defect_q0']
        if defect_q0_calc.is_finished_ok:
            self.ctx.defect_q0_energy = orm.Float(host_calc.outputs.output_parameters.get_dict()['E_KS'])
            self.ctx.defect_q0_VT = sisl.get_sile(calc_defect_q0.outputs.remote_folder.get_remote_path()+"/aiida.VT")
        else:
            self.report(
                'SIESTA for the defect structure (with charge 0) has failed with status {}'
                .format(defect_q0_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED

        # Defect (q=q)
        defect_q_calc = self.ctx['calc_defect_q']
        if defect_q_calc.is_finished_ok:
            self.ctx.defect_q_energy = orm.Float(defect_q_calc.outputs.output_parameters.get_dict()['E_KS']) # eV
            self.ctx.defect_q_VT = sisl.get_sile(calc_defect_q.outputs.remote_folder.get_remote_path()+"/aiida.VT")
        else:
            self.report(
                'SIESTA for the defect structure (with charge {}) has failed with status {}'
                .format(self.inputs.defect_charge.value,
                        defect_q_calc.exit_status))
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED



    def check_dft_potentials_gaussian_correction(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """

        # Host
        host_pp = self.ctx['pp_host']
        if host_pp.is_finished_ok:
            data_array = host_pp.outputs.output_data.get_array('data')
            v_data = orm.ArrayData()
            v_data.set_array('data', data_array)
            self.ctx.v_host = v_data
        else:
            self.report(
                'Post processing for the host structure has failed with status {}'
                .format(host_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED

        # Defect (q=0)
        defect_q0_pp = self.ctx['pp_defect_q0']
        if defect_q0_pp.is_finished_ok:
            data_array = host_pp.outputs.output_data.get_array('data')
            v_data = orm.ArrayData()
            v_data.set_array('data', data_array)
            self.ctx.v_defect_q0 = v_data
        else:
            self.report(
                'Post processing for the defect structure (with charge 0) has failed with status {}'
                .format(defect_q0_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED

        # Defect (q=q)
        defect_q_pp = self.ctx['pp_defect_q']
        if defect_q_pp.is_finished_ok:
            data_array = host_pp.outputs.output_data.get_array('data')
            v_data = orm.ArrayData()
            v_data.set_array('data', data_array)
            self.ctx.v_defect_q = v_data
        else:
            self.report(
                'Post processing for the defect structure (with charge {}) has failed with status {}'
                .format(self.inputs.defect_charge.value,
                        defect_q_pp.exit_status))
            return self.exit_codes.ERROR_PP_CALCULATION_FAILED