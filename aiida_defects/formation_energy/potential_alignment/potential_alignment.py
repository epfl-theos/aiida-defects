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
from aiida.common import AttributeDict
from aiida.engine import WorkChain, calcfunction, if_
from qe_tools import CONSTANTS

from .utils import get_interpolation
from .lany_zunger.lany_zunger import LanyZungerAlignmentWorkchain
from .density_weighted.density_weighted import DensityWeightedAlignmentWorkchain
from .mae.mae import MaeAlignmentWorkchain


valid_schemes = {
    'lany_zunger' : LanyZungerAlignmentWorkchain,
    'density_weighted': DensityWeightedAlignmentWorkchain,
    'mae': MaeAlignmentWorkchain
}

class PotentialAlignmentWorkchain(WorkChain):
    """
    Align two electrostatic potentials according to a specified method.
    """

    @classmethod
    def define(cls, spec):
        super(PotentialAlignmentWorkchain, cls).define(spec)
        spec.input('allow_interpolation',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="Whether to allow arrays of different shapes to be interpolated")
        spec.expose_inputs(DensityWeightedAlignmentWorkchain,
            namespace='density_weighted',
            namespace_options={'required': False, 'populate_defaults': False})
        spec.expose_inputs(MaeAlignmentWorkchain,
            namespace='mae',
            namespace_options={'required': False, 'populate_defaults': False})
        spec.expose_inputs(LanyZungerAlignmentWorkchain,
            namespace='lany_zunger',
            namespace_options={'required': False, 'populate_defaults': False})

        spec.outline(
            cls.setup,
            if_(cls.interpolation_required)(
                cls.do_interpolation,
            ),
            cls.calculate_alignment,
            cls.check_alignment_workchain,
            cls.results,
        )
        #spec.expose_outputs(PwBaseWorkChain, exclude=('output_structure',))
        spec.output('alignment_required', valid_type=orm.Float, required=True)

        # Exit codes
        spec.exit_code(201, 'ERROR_INPUT_BAD_SCHEME',
            message='the alignment scheme requested is unknown.')
        spec.exit_code(202, 'ERROR_INPUT_NO_SCHEME',
            message='no alignment scheme was setup.')
        spec.exit_code(203, 'ERROR_INPUT_EXTRA_ARRAYS',
            message='an ArrayData object has more than one array packed inside.')
        spec.exit_code(204, 'ERROR_INPUT_WRONG_SHAPE',
            message='all input arrays must have the same shape, unless interpolation is allowed.')
        spec.exit_code(205, 'ERROR_INPUT_WRONG_ASPECT_RATIO',
            message='all input arrays must have the same aspect ratio for interpolation to be effective.')
        spec.exit_code(301, 'ERROR_SUB_PROCESS_FAILED_INTERPOLATION',
            message='the interpolation could not be completed.')
        spec.exit_code(302, 'ERROR_SUB_PROCESS_FAILED_ALIGNMENT',
            message='the potential alignment could not be completed.')
        spec.exit_code(999, "ERROR_NOT_IMPLEMENTED",
            message="The requested method is not yet implemented.")

    def setup(self):
        """
        Input validation and context setup
        """

        # Only one namespace should be used at a time, and only one
        schemes_found = []
        for namespace in valid_schemes:
            if namespace in self.inputs:
                schemes_found.append(namespace)
        if len(schemes_found) == 1:
            self.ctx.alignment_scheme = namespace
        elif len(schemes_found) == 0:
            return self.exit_codes.ERROR_INPUT_NO_SCHEME
        else:
            return self.exit_codes.ERROR_INPUT_BAD_SCHEME

        # Collect the inputs from the selected scheme
        inputs = AttributeDict(
            self.exposed_inputs(valid_schemes[self.ctx.alignment_scheme],
            namespace=self.ctx.alignment_scheme))
        self.ctx.inputs = inputs


        # Array should have the same shape - if not they should be interpolated to have the same shape
        # Collect the arrays
        # arrays = {
        #     'first_potential': inputs.first_potential,
        #     'second_potential': inputs.second_potential
        # }
        arrays = {
            'first_potential': self.inputs[self.ctx.alignment_scheme]['first_potential'],
            'second_potential': self.inputs[self.ctx.alignment_scheme]['second_potential']
        }
        if 'charge_density' in inputs: # density-weighted case
            arrays['charge_density'] = inputs.charge_density

        # # Check if ArrayData objects have more than one array packed in them
        # for array in arrays.values():
        #     if len(array.get_arraynames()) != 1:
        #         return self.exit_codes.ERROR_INPUT_EXTRA_ARRAYS

        # Unpack and obtain the shapes
        array_shapes = {}
        for array_name, array in arrays.items():
            shape = array.get_shape(array.get_arraynames()[0])
            array_shapes[array_name] = shape

        # Check if the shapes are the same. If not, we must be allowed to interpolate
        self.ctx.interpolation_required = False
        if len(set(array_shapes.values())) != 1:
            self.ctx.interpolation_required = True
            if not self.inputs.allow_interpolation:
                return self.exit_codes.ERROR_INPUT_WRONG_SHAPE

        # For interpolation to be meaningful, the dimensions of the arrays must be compatible
        # For example, if one grid was (3,1) and another was (1,3), how would interpolation
        # be done? We try to avoid the situation where data is thrown away, and also one
        # where we make new grids which are the product of others.
        # Check that, when in ascending order according the dimension of the first axis,
        # all other axis keep the correct ordering.
        # If the reasoning was compelling, this could be relaxed later to the product type
        # situation where having a (3,1) and a (1,3) would result in a target grid of (3,3)

        sorted_shapes = sorted(list(array_shapes.values()))
        for index, shape in enumerate(sorted_shapes):
            for axis in [1,2]:   # Sorting is correct for axis 0, now check if the others are okay
                if index == 0:
                    continue
                else:
                    if (shape[axis] < sorted_shapes[index-1][axis]) :  # Are the values not ascending?
                        return self.exit_codes.ERROR_INPUT_WRONG_ASPECT_RATIO

        # Keep the dicts for later use
        self.ctx.arrays = arrays
        self.ctx.array_shapes = array_shapes


    def interpolation_required(self):
        """
        Return wether interpolation of the input arrays is needed due to their sizes being mismatched
        """
        return self.ctx.interpolation_required


    def do_interpolation(self):
        """
        If interpolation is required, apply it
        """
        self.report('Interpolating between mismatched arrays')

        shapes_array = np.array(list(self.ctx.array_shapes.values()))
        # Get max size along each axis - this is the target array size
        target_shape = orm.List(list=[
            np.max(shapes_array[:,0]),
            np.max(shapes_array[:,1]),
            np.max(shapes_array[:,2])
        ])

        self.report('Target interpolation size: {}'.format(target_shape.get_list()))

        self.report('Doing interpolation')
        interpolated_arrays = {}
        # for array_name, array in self.ctx.arrays.items():
        #     interpolated_arrays[array_name] = get_interpolation(
        #         input_array=array,
        #         target_shape=target_shape)
        # # Replace input arrays with interpolated versions
        # for array_name, array in interpolated_arrays.items():
        #     self.ctx.inputs[array_name] = array

        interpolated_arrays['first_potential'] = get_interpolation(
                input_array=self.inputs[self.ctx.alignment_scheme]['first_potential'],
                target_shape=target_shape)
        interpolated_arrays['second_potential'] = get_interpolation(
                input_array=self.inputs[self.ctx.alignment_scheme]['second_potential'],
                target_shape=target_shape)
        # Replace input arrays with interpolated versions
        for array_name, array in interpolated_arrays.items():
            self.ctx.inputs[array_name] = array

        return

    def calculate_alignment(self):
        """
        Calculate the alignment according to the requested scheme
        """
        # Get the correct alignment scheme workchain
        alignment_workchain = valid_schemes[self.ctx.alignment_scheme]

        inputs = self.ctx.inputs

        workchain_future = self.submit(alignment_workchain, **inputs)
        self.to_context(**{'alignment_wc': workchain_future})

        return


    def check_alignment_workchain(self):
        """
        Check if the model potential alignment workchain have finished correctly.
        If yes, assign the outputs to the context
        """

        alignment_workchain = self.ctx['alignment_wc']

        if not alignment_workchain.is_finished_ok:
            self.report(
                'Potential alignment workchain has failed with status {}'
                .format(alignment_workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ALIGNMENT
        else:
            self.ctx.alignment = alignment_workchain.outputs.alignment_required

    def results(self):
        """
        Collect results
        """
        self.report(
            "Completed alignment. An alignment of {} eV is required".format(self.ctx.alignment.value))
        self.out('alignment_required', self.ctx.alignment)
