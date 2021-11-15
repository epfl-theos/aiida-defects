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
from aiida_defects.formation_energy_siesta.utils import get_vbm_siesta_manual_bands , 
from aiida_defects.formation_energy_siesta.utils import output_energy_manual 
from aiida_defects.formation_energy_siesta.utils import output_total_electrons_manual
 
class FormationEnergyWorkchainSIESTAManual(FormationEnergyWorkchainBase):
    """
    Compute the formation energy for a given defect using SIESTA code
    """
    print("Formatioan Energy Worchain for Siesta Loaded")
    print("This WC For Neutral Systems with with Remote Folder Node")


    @classmethod
    def define(cls, spec):
        super(FormationEnergyWorkchainSIESTAManual, cls).define(spec)
        
        spec.input('host_remote',
                    valid_type = orm.RemoteData, 
                    required  =True) 
        spec.input('defect_neutral_remote',
                   valid_type = orm.RemoteData, 
                   required  = True) 


        #===================================================================
        # The Steps of Workflow
        #===================================================================

        spec.outline(cls.is_remote,
                     cls.setup,
                     cls.retrieving_dft_data,
                     cls.compute_no_corrected_formation_energy,  
                     )
                
                        
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
        if self.inputs.host_remote is not None:
            self.report("The Host Remote Provided")
            hr = True
        if self.inputs.defect_q0_remote is not None:
            self.report("The Defect q0 Remote Provided")
            dq0r = True
        if hr and dq0r:
            self.report("All Needed Remote Provided ")
        else:
            spec.ERROR_REMOTE_Folder
 
    
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

      



