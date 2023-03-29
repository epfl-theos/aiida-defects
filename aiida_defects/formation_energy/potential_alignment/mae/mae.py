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
from .utils import get_alignment, AllValuesMaskedError, convert_Hat_to_Ryd


class MaeAlignmentWorkchain(WorkChain):
    """
    Compute the alignment needed between two electrostatic potentials.
    Data points are included or excluded based on their distance from the defect site.
    The largest possible sphere

    The root mean squared difference between two potentials is computed using:
    \begin{equation}
    x =  \int \left| ( V_2 - V_1 + \Delta z ) \right|
    \end{equation}
    where:
        * V_1 and V_2 are the potentials to align
        * \Delta_z is the required alignment

    """

    @classmethod
    def define(cls, spec):
        super(MaeAlignmentWorkchain, cls).define(spec)
        spec.input('first_potential',
            valid_type=orm.ArrayData,
            help="The first electrostatic potential array")
        spec.input('second_potential',
            valid_type=orm.ArrayData,
            help="The second electrostatic potential array")
        spec.input("defect_site",
            valid_type=orm.List,
            help="Defect site position in crystal coordinates.")

        spec.outline(
            cls.setup,
            cls.compute_difference,
            cls.calculate_alignment,
            cls.results,
        )
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

        ### Temporary solution to convert potential to the same unit, has to be redone properly.
        ### The potentials generate by pp.x are in Rydberg while the model potential is in Hartree
        if len(self.inputs.second_potential.get_arraynames()) == 1:
            #v_model = orm.ArrayData()
            #v_model.set_array('data',self.inputs.second_potential.get_array(self.inputs.second_potential.get_arraynames()[0])*-2.0) # Hartree to Ry unit of potential - This is dirty - need to harmonise units
            v_model = convert_Hat_to_Ryd(self.inputs.second_potential)
        else:
            v_model = self.inputs.second_potential

        self.ctx.potential_difference = get_potential_difference(
            first_potential = self.inputs.first_potential,
            # second_potential = self.inputs.second_potential
            second_potential = v_model
        )


    def calculate_alignment(self):
        """
        Compute the alignment
        """
        try:
            self.ctx.alignment = get_alignment(
                potential_difference = self.ctx.potential_difference,
                defect_site= self.inputs.defect_site
            )
        except AllValuesMaskedError:
            return self.exit_codes.ERROR_ALL_VALUES_MASKED


    def results(self):
        """
        Pack the results
        """
        self.out('alignment_required', self.ctx.alignment)
        self.out('potential_difference', self.ctx.potential_difference)
