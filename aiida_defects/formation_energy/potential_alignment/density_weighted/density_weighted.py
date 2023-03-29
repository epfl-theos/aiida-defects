# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida import orm
from aiida.engine import WorkChain, calcfunction

from aiida_defects.formation_energy.potential_alignment.utils import get_potential_difference
from .utils import get_alignment, AllValuesMaskedError


class DensityWeightedAlignmentWorkchain(WorkChain):
    """
    Comput the alignment needed between two electrostatic potentials according to
    the charge-weighted potential alignment method.
    """

    @classmethod
    def define(cls, spec):
        super(DensityWeightedAlignmentWorkchain, cls).define(spec)
        spec.input('first_potential',
            valid_type=orm.ArrayData,
            help="The first electrostatic potential array")
        spec.input('second_potential',
            valid_type=orm.ArrayData,
            help="The second electrostatic potential array")
        spec.input('charge_density',
            valid_type=orm.ArrayData,
            help="The fitted model charge density array")
        spec.input('tolerance',
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0e-3),
            help="The threshold for determining whether a given array element has charge density present")

        spec.outline(
            cls.setup,
            cls.compute_difference,
            cls.calculate_alignment,
            cls.results,
        )
        #spec.expose_outputs(PwBaseWorkChain, exclude=('output_structure',))
        spec.output('alignment_required',
            valid_type=orm.Float,
            required=True,
            help="The computed potential alignment required")
        spec.output('potential_difference',
            valid_type=orm.ArrayData,
            required=True,
            help="The unmasked difference in electrostatic potentials")

        # Exit codes
        spec.exit_code(301, 'ERROR_ALL_VALUES_MASKED',
            message='All values in the potential difference array were masked. '
                'Try increasing the tolerance to include fewer elements from the charge density array.')


    def setup(self):
        pass


    def compute_difference(self):
        """
        Compute the difference of the  two potentials
        """

        self.ctx.potential_difference = get_potential_difference(
            first_potential = self.inputs.first_potential,
            second_potential = self.inputs.second_potential
        )


    def calculate_alignment(self):
        """
        Compute the alignment
        """

        try:
            self.ctx.alignment = get_alignment(
                potential_difference = self.ctx.potential_difference,
                charge_density = self.inputs.charge_density,
                tolerance = self.inputs.tolerance
            )
        except AllValuesMaskedError:
            return self.exit_codes.ERROR_ALL_VALUES_MASKED



    def results(self):
        """
        Pack the results
        """
        self.out('alignment_required', self.ctx.alignment)
        self.out('potential_difference', self.ctx.potential_difference)
