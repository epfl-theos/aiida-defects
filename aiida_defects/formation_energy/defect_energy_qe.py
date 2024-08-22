# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import

from aiida import orm
from aiida.engine import WorkChain, calcfunction, ToContext, if_, submit
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida_quantumespresso.common.types import RelaxType
from aiida_defects.formation_energy.defect_energy_base import DefectFormationEnergyWorkchainBase
from .utils import get_vbm

# pylint: disable=line-too-long

PwRelaxWorkChain = WorkflowFactory("quantumespresso.pw.relax")
PpCalculation = CalculationFactory("quantumespresso.pp")


class DefectEnergyWorkchainQE(DefectFormationEnergyWorkchainBase):
    """
    Compute the defect formation energy for a given defect using QuantumESPRESSO
    """

    _relax_type = {
        "fixed": RelaxType.NONE,
        "relax": RelaxType.POSITIONS,
        "vc-relax": RelaxType.POSITIONS_CELL,
    }

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # DFT and DFPT calculations with QuantumESPRESSO are handled with different codes, 
        # so here we keep track of things with two separate namespaces. An additional code, 
        # and an additional namespace, is used for postprocessing

        # spec.input_namespace(
        #     "qe.dft",
        #     help="Inputs for DFT calculations",
        # )
        spec.input_namespace(
            "qe.dft.supercell", 
            help="Inputs for DFT calculations on supercells"
        )
        spec.input_namespace(
            "qe.dft.unitcell",
            required=False,
            help="Inputs for a DFT calculation on an alternative host cell for use with DFPT",
        )
        spec.input_namespace(
            "qe.dfpt",
            help="Inputs for DFPT calculation for calculating the relative permittivity of the host material",
        )
        spec.input_namespace(
            "qe.pp",
            help="Inputs for postprocessing calculations"
        )

        # DFT inputs (PW.x)
        spec.input(
            "qe.dft.supercell.code",
            valid_type=orm.Code,
            help="The pw.x code to use for the calculations"
        )
        spec.input(
            "qe.dft.supercell.relaxation_scheme",
            valid_type=orm.Str,
            help="Option to relax the cell. Possible options are : ['fixed', 'relax']"
        )
        spec.input(
            "qe.dft.supercell.protocol",
            valid_type=orm.Str,
            help="Protocol for dft calculations. Possible options are: ['moderate', 'fast', 'precise']",
        )
        spec.input(
            "qe.dft.supercell.overrides",
            valid_type=orm.Dict,
            required=False,
            help="",
        )
        spec.input(
            "qe.dft.unitcell.protocol",
            valid_type=orm.Str,
            help="Protocol for dft calculations. Possible options are: ['moderate', 'fast', 'precise']",
        )
        spec.input(
            "qe.dft.unitcell.overrides",
            valid_type=orm.Dict,
            required=False,
            help="",
        )
        # spec.input(
        #     "",
        #     valid_type=orm.,
        #     help="",
        # )

        # Postprocessing inputs (PP.x)
        spec.input("qe.pp.code",
            valid_type=orm.Code,
            help="The pp.x code to use for the calculations")
        spec.input("qe.pp.scheduler_options",
            valid_type=orm.Dict,
            help="Scheduler options for the PP.x calculations")

        # DFPT inputs (PH.x)
        spec.input("qe.dfpt.code",
            valid_type=orm.Code,
            help="The ph.x code to use for the calculations")
        spec.input("qe.dfpt.scheduler_options",
            valid_type=orm.Dict,
            help="Scheduler options for the PH.x calculations")

        spec.outline(
            # cls.setup,
            cls.run_chemical_potential_workchain,
            cls.check_chemical_potential_workchain,
            cls.prep_dft_calcs_gaussian_correction,
            cls.check_dft_calcs_gaussian_correction
        )

    def prep_dft_calcs_gaussian_correction(self):
        """
        Get the required inputs for the Gaussian Countercharge correction workchain.
        This method runs the required calculations to generate the energies and potentials
        for the Gaussian scheme.
        """

        self.report("Setting up the Gaussian Countercharge correction workchain")

        # Host structure
        inputs = PwRelaxWorkChain.get_builder_from_protocol(
            code=self.inputs.qe.dft.supercell.code,
            structure=self.inputs.host_structure,
            protocol=self.inputs.qe.dft.supercell.protocol,
            overrides=self.inputs.qe.dft.supercell.overrides,
            relax_type=RelaxType.POSITIONS
        )
        # inputs = PwBaseWorkChain(*args, overrides=None)
        future = self.submit(PwRelaxWorkChain, **inputs)
        self.report(
            "Launching PWSCF for the host structure (PK={}) with charge {} (PK={})".format(
                self.inputs.host_structure.pk, "0.0", future.pk
            )
        )
        self.to_context(**{"calc_host": future})

    def check_dft_calcs_gaussian_correction(self):
        """
        Check if the required calculations for the Gaussian Countercharge correction workchain
        have finished correctly.
        """

        # Host
        # if self.inputs.run_pw_host:
        host_calc = self.ctx["calc_host"]
        if host_calc.is_finished_ok:
            self.ctx.host_energy = orm.Float(
                host_calc.outputs.output_parameters.get_dict()["energy"]
            )  # eV
            self.report(
                "The energy of the host is: {} eV".format(
                    self.ctx.host_energy.value
                )
            )
            # self.ctx.host_vbm = orm.Float(host_calc.outputs.output_band.get_array('bands')[0][-1]) # valence band maximum
            self.ctx.host_vbm = orm.Float(get_vbm(host_calc))
            self.report(
                "The top of valence band is: {} eV".format(self.ctx.host_vbm.value)
            )
            is_insulator, band_gap = orm.nodes.data.array.bands.find_bandgap(
                host_calc.outputs.output_band
            )
            if not is_insulator:
                self.report(
                    "WARNING! The ground state of the host structure is metallic!"
                )
        else:
            self.report(
                "PWSCF for the host structure has failed with status {}".format(
                    host_calc.exit_status
                )
            )
            return self.exit_codes.ERROR_DFT_CALCULATION_FAILED