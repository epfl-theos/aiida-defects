# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import


def run_pw_calculation(pw_inputs, charge, run_type, additional_inputs):
    """
    Run a QuantumESPRESSO PW.x calculation by invoking the PW workchain 

    Parameters
    ----------
    pw_inputs : AiiDA Dict
        A dictionary containing parameters for the PW calculation
    charge : AiiDA Float
        The required total system charge. Adding an electron is negative by convention
    run_type: AiiDA String
        The desired type of calculation. Allowed values: 'scf', 'relax', 'vc-relax'

    Returns
    -------
    pw_object?
        A future representing the submitted calculation
    """

    required_keys = [
        'code', 'pseudo_family', 'parameters', 'settings', 'options',
        'structure'
    ]

    # Validate input dictionary
    for key in required_keys:
        if key not in pw_inputs:
            raise KeyError(
                "Required key, '{}' not found in input dictionary".format(key))

    # Validate 'run_type'
    if run_type not in ['scf', 'relax', 'vc-relax']:
        raise ValueError("Run type, '{}', not recognised".format(run_type))

    pw_inputs['parameters']['SYSTEM']['tot_charge'] = charge

    if run_type == 'relax' or run_type == 'vc-relax':
        pw_inputs['relaxation_scheme'] = run_type

        # additional_inputs=[
        #     'vdw_table',
        #     'final_scf',
        #     'group',
        #     'max_iterations',
        #     'max_meta__convergence_iterations',
        #     'meta_convergence',
        #     'volume_convergence',
        #     'clean_workdir'
        # ]

        # for item in additional_inputs:
        #     if item in pw_inputs:

        running = submit(PwRelaxWorkChain, **inputs)
        self.report(
            'Launching PwRelaxWorkChain for structure, {}, with charge {} (PK={})'
            .format(pw_inputs.structure, charge, running.pid))
        return running

    else:
        running = submit(PwBaseWorkChain, **inputs)
        self.report(
            'Launching PwBaseWorkChain for structure, {}, with charge {} (PK={})'
            .format(pw_inputs.structure, charge, running.pid))
        return running
