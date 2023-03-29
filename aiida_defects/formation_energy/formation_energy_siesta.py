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
from aiida.plugins import WorkflowFactory
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_defects.formation_energy.formation_energy_base import FormationEnergyWorkchainBase
from aiida_defects.formation_energy.utils import run_pw_calculation
from .utils import get_raw_formation_energy, get_corrected_formation_energy, get_corrected_aligned_formation_energy


class FormationEnergyWorkchainSiesta(FormationEnergyWorkchainBase):
    """
    Compute the formation energy for a given defect using Siesta
    """
    @classmethod
    def define(cls, spec):
        super(FormationEnergyWorkchainSiesta, cls).define(spec)

        # Namespace to make it clear which code is being used.
        spec.input_namespace('siesta.dft.supercell',
            help="Inputs for DFT calculations on supercells")

        # DFT inputs (Siesta)
        # spec.input("siesta.dft.supercell.code",
        #     valid_type=orm.Code,
        #     help="The Siesta code to use for the supercell calculations")
        # spec.input("siesta.dft.supercell.parameters",
        #     valid_type=orm.Dict,
        #     help="Parameters for the supercell calculations. Some will be set automatically")
        # spec.input("siesta.dft.supercell.scheduler_options",
        #     valid_type=orm.Dict,
        #     help="Scheduler options for the Siesta calculation")

        spec.outline(
            cls.setup,
            if_(cls.correction_required)(
                if_(cls.is_gaussian_scheme)(
                    cls.placeholder,
                    cls.run_gaussian_correction_workchain),
                if_(cls.is_point_scheme)(
                    cls.raise_not_implemented
                    #cls.prepare_point_correction_workchain,
                    #cls.run_point_correction_workchain),
                ),
                cls.check_correction_workchain),
            cls.compute_formation_energy
        )

    def placeholder(self):
        """
        Placeholder method
        """
        pass