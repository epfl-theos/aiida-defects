# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida import orm
from aiida.engine import WorkChain, calcfunction
from aiida_defects.formation_energy.potential_alignment.lany_zunger import lany_zunger


@calcfunction
def testing():
    return orm.Float(0.0)


class PotentialAlignmentWorkchain(WorkChain):
    """
    Align two electrostatic potentials according to a specified method.
    """

    @classmethod
    def define(cls, spec):
        super(PotentialAlignmentWorkchain, cls).define(spec)
        spec.input('first_potential', valid_type=orm.ArrayData)
        spec.input('second_potential', valid_type=orm.ArrayData)
        spec.input(
            'alignment_scheme',
            valid_type=orm.Str,
            default=orm.Str('lany-zunger'))
        spec.input('interpolate', valid_type=orm.Bool, default=orm.Bool(False))
        spec.outline(
            cls.setup,
            cls.do_interpolation,
            cls.calculate_alignment,
            cls.results,
        )
        #spec.expose_outputs(PwBaseWorkChain, exclude=('output_structure',))
        spec.output('alignment_required', valid_type=orm.Float, required=True)
        # Exit codes
        spec.exit_code(
            401,
            'ERROR_SUB_PROCESS_FAILED_WRONG_SHAPE',
            message=
            'the two electrostatic potentials must be the same shape, unless interpolation is allowed'
        )
        spec.exit_code(
            402,
            'ERROR_SUB_PROCESS_FAILED_INTERPOLATION',
            message='the interpolation could not be completed')
        spec.exit_code(
            403,
            'ERROR_SUB_PROCESS_FAILED_BAD_SCHEME',
            message='the alignment scheme requested is unknown')

    def setup(self):
        """
        Input validation and context setup
        """
        # Two potentials need have the same shape
        first_potential_shape = self.inputs.first_potential.get_shape(
            self.inputs.first_potential.get_arraynames()[0])
        second_potential_shape = self.inputs.second_potential.get_shape(
            self.inputs.second_potential.get_arraynames()[0])
        if first_potential_shape != second_potential_shape:
            if self.inputs.interpolate:
                self.ctx.interpolation_required = True
            else:
                self.report(
                    'The two potentials could not be aligned as they are the different shapes and interpolation is not allowed.'
                )
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WRONG_SHAPE
        else:
            self.ctx.interpolation_required = False

        if self.inputs.alignment_scheme not in ['lany-zunger']:
            self.report(
                'The requested alignment scheme, "{}" is not recognised.'.
                format(self.inputs.alignment_scheme))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BAD_SCHEME

        self.ctx.first_potential = self.inputs.first_potential
        self.ctx.second_potential = self.inputs.second_potential

        self.ctx.alignment = 0.0

    def do_interpolation(self):
        """ 
        If interpolation is required, apply it
        """

        if self.ctx.interpolation_required:
            self.report('Doing potential alignment')
            # TODO: Call the interpolation function and update the context

        return

    def calculate_alignment(self):
        """
        Calculate the alignment according to the requested scheme
        """
        # Call the correct alignment scheme
        if self.inputs.alignment_scheme == 'lany-zunger':
            # TODO: import and call the alignment function
            self.ctx.alignment = testing()
            #self.ctx.alignment =

        return

    def results(self):
        """
        Collect results
        """
        self.report(
            "Completed alignment. An alignment of {} eV is required".format(
                self.ctx.alignment.value))
        self.out('alignment_required', self.ctx.alignment)
