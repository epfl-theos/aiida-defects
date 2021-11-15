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
from aiida_defects.formation_energy_siesta.formation_energy_base import FormationEnergyWorkchainBase
from aiida_defects.formation_energy_siesta.utils import get_raw_formation_energy,
from aiida_defects.formation_energy_siesta.utils import get_corrected_formation_energy
from aiida_defects.formation_energy_siesta.utils import get_corrected_aligned_formation_energy 

import sisl
from aiida_defects.formation_energy_siesta.utils import get_vbm_siesta_manual_bands 
from aiida_defects.formation_energy_siesta.utils import output_energy_manual
from aiida_defects.formation_energy_siesta.utils import output_total_electrons_manual
        
 
class FormationEnergyWorkchainSIESTAManual(FormationEnergyWorkchainBase):
    """
    Compute the formation energy for a given defect using SIESTA code
    """
    print("Formatioan Energy Worchain for Siesta Loaded")
    print("This WC is For Charged Systems with Remote Folder Node")


    @classmethod
    def define(cls, spec):
        super(FormationEnergyWorkchainSIESTAManual, cls).define(spec)

        spec.input('host_remote',
                    valid_type = orm.RemoteData, 
                    required  =True) 
        spec.input('defect_neutral_remote',
                   valid_type = orm.RemoteData, 
                   required  = True) 
        spec.input('defect_charged_remote',
                   valid_type = orm.RemoteData, 
                   required  = True) 

        #===================================================================
        # What calculations to run (SIESTA) 
        #===================================================================

        #===================================================================
        # The Steps of Workflow
        #===================================================================

        spec.outline(cls.is_remote,
                     cls.is_calculate_rho, 
                     cls.setup,
                     cls.retrieving_dft_data,
                     cls.retrieving_dft_potentials,
                     if_(cls.is_retrieving_dft_rhos)(
                         cls.retrieving_dft_rhos),
                     if_(cls.correction_required)(
                         if_(cls.is_gaussian_model_scheme)(
                             cls.run_gaussian_model_correction_workchain,
                             cls.check_gaussian_model_correction_workchain,
                             cls.compute_formation_energy_gaussian_model
                        ).else_(cls.compute_no_corrected_formation_energy)
                         ))
                
                #cls.run_dft_calcs,
                #if_(cls.correction_required)(
                #    if_(cls.is_gaussian_model_scheme)(
                #        cls.check_dft_potentials_gaussian_correction,
                #        cls.run_gaussian_model_correction_workchain,
                #        cls.check_gaussian_model_correction_workchain,
                #        cls.compute_formation_energy_gaussian_model
                #        ),
                #    if_(cls.is_gaussian_rho_scheme)(
                #        cls.check_dft_potentials_gaussian_correction,
                #        #cls.run_gaussian_rho_correction_workchain,
                #        cls.raise_not_implemented,
                #        )),

                    #) 
                        
        spec.exit_code(201,'ERROR_REMOTE_Folder',
            message='The remote folder failed')


    #======================================================
    # Remote Folder Parts
    #======================================================
    def is_remote(self):
        """
        """
        hr = False
        dq0r = False
        #dqr = False
        if self.inputs.host_remote is not None:
            self.report("The Host Remote Provided")
            hr = True
        else:
            spec.ERROR_REMOTE_Folder

        if self.inputs.defect_q0_remote is not None:
            self.report("The Defect q0 Remote Provided")
            dq0r = True
        else:
            spec.ERROR_REMOTE_Folder
        if self.inputs.defect_q_remote is not None:
            self.report("The Defect q Remote Provided")
            dqr = True
        else:
            spec.ERROR_REMOTE_Folder
        if hr and dq0r and dqr:
            self.report("All Needed Remote Provided ")
        else:
            spec.ERROR_REMOTE_Folder
 
   
    def is_calculate_rho(self):
        """
        """
        if self.inputs.correction_scheme == 'rho' or self.inputs.correction_scheme == 'gaussian-rho':
            self.report("Will calculate DFT RHO")
            self.is_rho = True
        else:
            self.is_rho =  False

    def is_retrieving_dft_rhos(self):
        ""
        ""
        if self.is_rho:
            return True
        else:
            return False


    #=======================================================
    # Retrieving DFT Calculations for None Correction Scheme  
    #=======================================================


    def retrieving_dft_data(self):
        """
        """

        #----- 
        # Host
        #-----
        host = self.inputs.host_remote
        host_fdf_file = sisl.get_sile(host.get_remote_path()+"/input.fdf")
        host_label = host_fdf_file.get("SystemLabel")
        self.report(f"DEBUG: The Host System Label: {host_label}")
        self.ctx.host_energy = orm.Float(output_energy_manual(host))
        self.report(f"The Energy of the Host is: {self.ctx.host_energy.value} eV")

        #---------------
        # Defect Neutral
        #---------------

        defect_q0 = self.inputs.defect_neutral_remote
        defect_q0_fdf_file = sisl.get_sile(defect_q0.get_remote_path()+"/input.fdf")
        defect_q0_label = defect_q0_fdf_file.get("SystemLabel")
        self.report(f"DEBUG: The Defect q0 System Label: {defect_q0_label}")
        self.ctx.defect_q0_energy = orm.Float(output_energy_manual(defect_q0))
        self.report(f'The Energy of the Defect q0 is: {self.ctx.defect_q0_energy.value} eV')

        #---------------
        # Defect Charged
        #---------------

        defect_q = self.inputs.defect_charged_remote
        defect_q_fdf_file = sisl.get_sile(defect_q.get_remote_path()+"/input.fdf")
        defect_q_label = defect_q_fdf_file.get("SystemLabel")
        self.report(f"DEBUG: The Defect q System Label: {defect_q_label}")
        self.ctx.defect_q_energy = orm.Float(output_energy_manual(defect_q))
        self.report(f'The Energy of the Defect q is: {self.ctx.defect_q_energy.value} eV')

    def retrieving_dft_potentials(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """
        self.report("The Charge is :{}".format(self.inputs.defect_charge))
         
        #----- 
        # Host
        #-----
        host = self.inputs.host_remote
        host_fdf_file = sisl.get_sile(host.get_remote_path()+"/input.fdf")
        host_label = host_fdf_file.get("SystemLabel")
        self.report(f"DEBUG: The Host System Label: {host_label}")
        data_array = sisl.get_sile(host.get_remote_path() +"/" + host_label + ".VT")
        vt = data_array.read_grid()  
        vt_data = orm.ArrayData()
        vt_data_grid = orm.ArrayData()
        vt_data.set_array('data', vt.grid) # The array
        vt_data_grid.set_array('grid',np.array(vt.shape)) # The Grid Shape
        self.ctx.v_host = vt_data
        self.ctx.v_host_grid = vt_data_grid
        self.report("DEBUG: The host VT file Read!")
        self.report(f"DEBUG: VT Array = {self.ctx.v_host}")
        self.report(f"DEBUG: VT Array Shape = {self.ctx.v_host_grid.get_array('grid')}")
        self.number_of_electrons = output_total_electrons_manual(host)
        self.report(f"DEBUG: Total Number of ELectrons in Host : {self.number_of_electrons}")
        self.ctx.host_vbm = orm.Float(get_vbm_siesta_manual_bands(host,host_label,self.number_of_electrons))
        self.report(f"DEBUG: VBM = {self.ctx.host_vbm.value}") 
 
        #---------------
        # Defect Neutral
        #---------------

        defect_q0 = self.inputs.defect_neutral_remote
        defect_q0_fdf_file = sisl.get_sile(defect_q0.get_remote_path()+"/input.fdf")
        defect_q0_label = defect_q0_fdf_file.get("SystemLabel")
        self.report(f"DEBUG: The Defect q0 System Label: {defect_q0_label}")

        data_array_q0 = sisl.get_sile(defect_q0.get_remote_path() +"/" + defect_q0_label + ".VT")
        vt_q0 = data_array_q0.read_grid()
        vt_q0_data = orm.ArrayData()
        vt_q0_grid_data = orm.ArrayData()
        vt_q0_data.set_array('data', vt_q0.grid)
        vt_q0_grid_data.set_array('grid', np.array(vt_q0.shape))
        self.ctx.v_defect_q0 = vt_q0_data
        self.ctx.v_defect_q0_grid = vt_q0_grid_data 
        self.report("DEBUG: The defect_q0 VT file Read!")
        self.report(f"DEBUG: VT q0 Array = {self.ctx.v_defect_q0}")
        self.report(f"DEBUG: VT q0 Array Shape = {self.ctx.v_defect_q0_grid.get_array('grid')}")

        #---------------
        # Defect Charged
        #---------------

        defect_q = self.inputs.defect_charged_remote
        defect_q_fdf_file = sisl.get_sile(defect_q.get_remote_path()+"/input.fdf")
        defect_q_label = defect_q_fdf_file.get("SystemLabel")
        self.report("DEBUG: The Defect q System Label: {}".format(defect_q_label) )

        data_array_q = sisl.get_sile(defect_q.get_remote_path() +"/" + defect_q_label + ".VT")
        vt_q = data_array_q.read_grid()
        vt_q_data = orm.ArrayData()
        vt_q_grid_data = orm.ArrayData()
        vt_q_data.set_array('data', vt_q.grid)
        vt_q_grid_data.set_array('grid', np.array(vt_q.shape))
        self.ctx.v_defect_q = vt_q_data
        self.ctx.v_defect_q_grid = vt_q_grid_data 
        self.report("DEBUG: The defect_q VT file Read!")
        self.report(f"DEBUG: VT q Array = {self.ctx.v_defect_q}")
        self.report(f"DEBUG: VT q Array Shape = {self.ctx.v_defect_q_grid.get_array('grid')}")


    def retrieving_dft_rhos(self):
        """
        Obtain The Charge Density from the SIESTA calculations.
        """
        import sisl
        
        # Host

        if self.inputs.run_siesta_host:
            host_node = self.ctx['calc_host']
            if host_node.is_finished_ok:
                data_array = sisl.get_sile(host_node.outputs.remote_folder.get_remote_path()+"/aiida.RHO")
                rho = data_array.read_grid()
                rho_data = orm.ArrayData()
                rho_data.set_array('data', rho.grid)
                self.ctx.rho_host = rho_data
                self.report("DEBUG: The host Rho file Read!")
                self.report("DEBUG: Rho Array = "+str(self.ctx.rho_host ))
            else:
                self.report('Post processing for the host structure has failed with status {}'.format(host_node.exit_status))
                return self.exit_codes.ERROR_PP_CALCULATION_FAILED
        else:
            HostNode = orm.load_node(self.inputs.host_node.value)
            self.report('Extracting SIESTA VT for host structure with charge {} from node PK={}'.format("0.0", self.inputs.host_node.value))
            data_array = sisl.get_sile(HostNode.outputs.remote_folder.get_remote_path()+"/aiida.VT")
            rho=data_array.read_grid()
            rho_data = orm.ArrayData()
            rho_data.set_array('data', rho.grid)
            self.ctx.rho_host = rho_data
            self.report("DEBUG: The host VT file Read!")
            self.report("DEBUG: VT Array = "+str(self.ctx.rho_host ))

         
        if self.inputs.run_siesta_defect_q0:
            defect_q0_rho = self.ctx['calc_host']
            if defect_q0_rho.is_finished_ok:
                data_array = sisl.get_sile(defect_q0_rho.outputs.remote_folder.get_remote_path()+"/aiida.RHO")
                rho = data_array.read_grid()
                rho_data = orm.ArrayData()
                rho_data.set_array('data', rho.grid)
                self.ctx.rho_host = rho_data
                self.report("DEBUG: The host Rho file Read!")
                self.report("DEBUG: Rho Array = "+str(self.ctx.rho_host ))
            else:
                self.report('Post processing for the host structure has failed with status {}'.format(defect_q0_rho.exit_status))
                return self.exit_codes.ERROR_PP_CALCULATION_FAILED
        else:
            Defect_q0Node = orm.load_node(self.inputs.defect_q0_node.value)
            self.report('Extracting SIESTA VT for host structure with charge {} from node PK={}'.format("0.0", self.inputs.defect_q0_node.value))
            data_array = sisl.get_sile(Defect_q0Node.outputs.remote_folder.get_remote_path()+"/aiida.VT")
            rho=data_array.read_grid()
            rho_data = orm.ArrayData()
            rho_data.set_array('data', rho.grid)
            self.ctx.rho_host = rho_data
            self.report("DEBUG: The host VT file Read!")
            self.report("DEBUG: VT Array = "+str(self.ctx.rho_host ))


