# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import
import sys

import numpy as np
import pymatgen

from aiida.engine import WorkChain, calcfunction, ToContext, while_, if_
from aiida import orm
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from . import utils
import six
from six.moves import range
from six.moves import zip

######TODO: Apply create suitable input before submitting the PW calc


class BandFillingWorkChain(WorkChain):
    """
    Workchain to compute band filling corrections
    bands_NiO=run(PwBandsWorkChain,code=Code.get_from_string(codename), structure = s, pseudo_family=Str(pseudo_family),
              options=ParameterData(dict=options), settings=ParameterData(dict=settings), kpoints_mesh=kpoints,
              parameters=parameters, optimization=Bool(False), relax=relax)
    """

    @classmethod
    def define(cls, spec):
        super(BandFillingWorkChain, cls).define(spec)
        spec.input("code", valid_type=orm.Code)
        spec.input("host_structure", valid_type=orm.StructureData)
        spec.input("defect_structure", valid_type=orm.StructureData)
        spec.input('options', valid_type=orm.Dict)
        spec.input("settings", valid_type=orm.Dict)
        spec.input('host_parameters', valid_type=orm.Dict, required=False)
        spec.input('defect_parameters', valid_type=orm.Dict, required=False)
        spec.input('pseudo_family', valid_type=orm.Str)
        spec.input('kpoints_mesh', valid_type=orm.KpointsData, required=False)
        spec.input(
            'kpoints_distance', valid_type=orm.Float, default=orm.Float(0.2))
        spec.input(
            'potential_alignment', valid_type=orm.Float, default=orm.Float(0.))
        spec.input(
            'skip_relax',
            valid_type=orm.Bool,
            required=False,
            default=orm.Bool(True))
        spec.input_group('relax')
        spec.input('host_bandstructure', valid_type=orm.Node, required=False)
        spec.input('defect_bandstructure', valid_type=orm.Node, required=False)
        spec.outline(
            if_(cls.should_run_host)(cls.run_host),
            if_(cls.should_run_defect)(cls.run_defect),
            cls.compute_band_filling, cls.retrieve_bands)
        spec.dynamic_output()

    def should_run_host(self):
        return not 'host_bandstructure' in self.inputs

    def run_host(self):
        if 'host_parameters' not in self.inputs:
            self.abort_nowait(
                'The host bandstructure calculation was requested but the "host parameters" dictionary\
            was not provided')

        inputs = {
            'code': self.inputs.code,
            'structure': self.inputs.host_structure,
            'options': self.inputs.options,
            'parameters': self.inputs.host_parameters,
            'settings': self.inputs.settings,
            'pseudo_family': self.inputs.pseudo_family,
        }

        if 'skip_relax' in self.inputs:
            inputs['relax'] = self.inputs.relax
        if 'kpoints_mesh' in self.inputs:
            inputs['kpoints_mesh'] = self.inputs.kpoints_mesh

        running = submit(PwBandsWorkChain, **inputs)
        self.report(
            'Launching the PwBandsWorkChain for the host with PK'.format(
                running.pid))
        return ToContext(host_bandsworkchain=running)

    def should_run_defect(self):
        return not 'defect_bandstructure' in self.inputs

    def run_defect(self):
        if 'defect_parameters' not in self.inputs:
            self.abort_nowait(
                'The defect bandstructure calculation was requested but the "host parameters" dictionary\
            was not provided')

        inputs = {
            'code': self.inputs.code,
            'structure': self.inputs.defect_structure,
            'options': self.inputs.options,
            'parameters': self.inputs.defect_parameters,
            'settings': self.inputs.settings,
            'pseudo_family': self.inputs.pseudo_family,
        }

        if 'skip_relax' in self.inputs:
            inputs['relax'] = self.inputs.relax
        if 'kpoints_mesh' in self.inputs:
            inputs['kpoints_mesh'] = self.inputs.kpoints_mesh

        running = submit(PwBandsWorkChain, **inputs)
        #print "PK", running.pk
        self.report(
            'Launching the PwBandsWorkChain for the defect with PK'.format(
                running.pid))
        return ToContext(defect_bandsworkchain=running)

    def compute_band_filling(self):
        if 'host_bandstructure' in self.inputs:
            host_bandstructure = self.inputs.host_bandstructure.out
        else:
            host_bandstructure = self.ctx.host_bandsworkchain.out
        if 'defect_bandstructure' in self.inputs:
            defect_bandstructure = self.inputs.defect_bandstructure.out
        else:
            defect_bandstructure = self.ctx.defect_bandsworkchain.out

        self.ctx.band_filling = bandfilling_ms_correction(
            host_bandstructure, defect_bandstructure,
            float(self.inputs.potential_alignment))

    def retrieve_bands(self):
        """
        Attach the relevant output nodes from the band calculation to the workchain outputs
        for convenience
        """
        self.report('BandFillingCorrection workchain succesfully completed')

        self.report(
            'The computed band filling correction is <{}> and <{}> eV for a donor and an acceptor, respectively'
            .format(self.ctx.band_filling['E_donor'],
                    self.ctx.band_filling['E_acceptor']))

        for label, value in six.iteritems(self.ctx.band_filling):
            self.out(str(label), orm.Float(value))


#         for link_label in ['primitive_structure', 'seekpath_parameters', 'scf_parameters', 'band_parameters', 'band_structure']:
#             if link_label in self.ctx.defect_bandsworkchain.out:
#                 node = self.ctx.workchain_bands.get_outputs_dict()[link_label]
#                 self.out('defect'+str(link_label), node)
#                 self.report("attaching {}<{}> as an output node with label '{}'"
#                     .format(node.__class__.__name__, node.pk, link_label))
#             if link_label in self.ctx.host_bandsworkchain.out:
#                 node = self.ctx.workchain_bands.get_outputs_dict()[link_label]
#                 self.out('host'+str(link_label), node)
#                 self.report("attaching {}<{}> as an output node with label '{}'"
#                     .format(node.__class__.__name__, node.pk, link_label))
