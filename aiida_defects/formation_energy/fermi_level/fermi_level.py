# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida.orm import Float, Int, Str, List, Bool, Dict, ArrayData, XyData, StructureData
from aiida.engine import WorkChain, calcfunction, ToContext, while_
import sys
import numpy as np
from scipy.optimize.nonlin import NoConvergence
from pymatgen.core.composition import Composition

from .utils import *

class FermiLevelWorkchain(WorkChain):
    '''
    Compute the self-consistent Fermi level by imposing the overall charge neutrality
    Here we implement method similar to Buckeridge et al., (doi:10.1016/j.cpc.2019.06.017)
    '''
    @classmethod
    def define(cls, spec):
        super(FermiLevelWorkchain, cls).define(spec)
        spec.input("defect_data", valid_type=Dict)
        spec.input("chem_potentials", valid_type=Dict)
        spec.input("temperature", valid_type=Float)
        spec.input("valence_band_maximum", valid_type=Float)
        spec.input("number_of_electrons", valid_type=Float, help="number of electrons in the unitcell used to compute the DOS")
        spec.input("unitcell", valid_type=StructureData)
        spec.input("DOS", valid_type=XyData)
        spec.input("band_gap", valid_type=Float)
        spec.input("dopant", valid_type=Dict, default=lambda: Dict(dict=None),
                help="aliovalent dopants specified by its charge and concentration. Used to compute the change in the defect concentrations with frozen defect approach")
        spec.input("tolerance_factor", valid_type=Float, default=lambda: Float(1e-10),
                help="tolerance factor use in the non-linear solver to solve for the self-consistent fermi level")

        spec.outline(
            cls.setup,
            cls.compute_sc_fermi_level,
        )
        spec.output('fermi_level', valid_type=ArrayData) # we use ArrayData instead of Float in other to be general and be able to accomodate the situtation where the chemical potential is a numpy array allowing to vectorize the calculations of defect concentrations in stability region instead of doing one value of chemical potential at a time.

        spec.exit_code(701, "ERROR_FERMI_LEVEL_FAILED",
            message="The number of electrons obtained from the integration of DOS is different from the expected number of electrons in the input"
        )
        spec.exit_code(702, "ERROR_NON_LINEAR_SOLVER_FAILED",
            message="The non-linear solver used to solve for the self-consistent Fermi level failed. The tolerance factor might be too small"
        )

    def setup(self):
        """
        Setup the calculation
        """
        chempot_dict = self.inputs.chem_potentials.get_dict()
#        for key, value in chempot_dict.items():
#            data_array = np.ones_like(value)
#            #print(data_array)
#            v_data = ArrayData()
#            v_data.set_array('data', data_array)
#            self.ctx.input_chem_shape = v_data

        # extracting the DOS of the unitcell, assuming that the calculation is non-spin polarized.
        dos_x = self.inputs.DOS.get_x()[1] - self.inputs.valence_band_maximum.value # Shift the top of valence band to zero
        v_data = ArrayData()
        v_data.set_array('data', dos_x)
        self.ctx.dos_x = v_data

        dos_y = self.inputs.DOS.get_y()[1][1]
        v_data = ArrayData()
        v_data.set_array('data', dos_y)
        self.ctx.dos_y = v_data

        mask = (dos_x <= 0.05)
        N_electron = np.trapz(dos_y[mask], dos_x[mask])
        if np.absolute(N_electron-self.inputs.number_of_electrons.value) > 5e-3:
            self.report('The number of electrons obtained from the integration of DOS is: {}'.format(N_electron))
            self.report('The number of electrons obtained from the integration of DOS is different from the expected number of electrons in the input')
            return self.exit_codes.ERROR_FERMI_LEVEL_FAILED

        #is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(unitcell_node.outputs.output_band)
        #if not is_insulator:
            #self.report('WARNING!')
            #self.report('The compound is metallic!')

    def compute_sc_fermi_level(self):
        try:
            E_Fermi = solve_for_sc_fermi(self.inputs.defect_data,
                                        self.inputs.chem_potentials,
                                        #self.ctx.input_chem_shape,
                                        self.inputs.temperature,
                                        self.inputs.unitcell,
                                        self.inputs.band_gap,
                                        self.ctx.dos_x,
                                        self.ctx.dos_y,
                                        self.inputs.dopant,
                                        self.inputs.tolerance_factor)

            self.ctx.sc_fermi_level = E_Fermi
            self.out('fermi_level', E_Fermi)
            self.report('The self-consistent Fermi level is: {} eV'.format(E_Fermi.get_array('data')))
        except NoConvergence:
            self.report("The non-linear solver used to solve for the self-consistent Fermi level failed. The tolerance factor might be too small")
            return self.exit_codes.ERROR_NON_LINEAR_SOLVER_FAILED
