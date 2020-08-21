# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/ConradJohnston/aiida-defects #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import absolute_import
from __future__ import print_function
from aiida.work.workchain import WorkChain
from aiida_defects.pp.pp import PpWorkChain
from aiida_defects.pp.fft_tools import planar_average, read_grid, differentiator
from six.moves import range


class SlabEpsilonSawtoothWorkChain(WorkChain):
    """
    WorkChain to calculate the dielectric constant of a slab structure within the Finite Electric Field approach 
    and a sawtooth potential. Both low and high frequency dielectric constant value can be obtained
    """
    """
    TODO:
    -update dielectric constant profile
    -dielectric profile function should be workfunction. this would require to change the output 
    of the planar average function
    - change to compute low_frequency. this require cvhanging the Ppworkchain commenting the line for scf
      and update structure                                                                                                                                                                                         
    """

    @classmethod
    def define(cls, spec):
        super(SlabEpsilonSawtoothWorkChain, cls).define(spec)
        spec.input("structure", valid_type=StructureData)
        spec.input("code_pw", valid_type=Str, required=False)
        spec.input("code_pp", valid_type=Str, required=False)
        spec.input("pseudo_family", valid_type=Str, required=False)
        spec.input('options', valid_type=ParameterData)
        spec.input("settings", valid_type=ParameterData)
        spec.input("kpoints", valid_type=KpointsData, required=False)
        spec.input('parameters', valid_type=ParameterData, required=False)
        spec.input(
            'magnetic_phase',
            valid_type=Str,
            required=False,
            default=Str('NM'))
        spec.input('B_atom', valid_type=Str, required=False)
        spec.input(
            'hubbard_u',
            valid_type=ParameterData,
            required=False,
            default=ParameterData(dict={}))
        spec.input(
            'epsilon_type',
            valid_type=Str,
            required=False,
            default=Str('high-frequency'))
        spec.input(
            'eamp', valid_type=Float, required=False, default=Float(0.001))
        spec.input(
            'eopreg', valid_type=Float, required=False, default=Float(0.01))
        spec.input(
            'emaxpos', valid_type=Float, required=False, default=Float(0.75))
        spec.input('edir', valid_type=Int, required=False, default=Int(3))
        spec.input(
            'sigma', valid_type=Float, required=False, default=Float(0.5))
        spec.outline(
            cls.initializing_e0,
            cls.run_ppworkchain_e0,
            cls.initializing_e1,
            cls.run_ppworkchain_e1,
            cls.initializing_e1_saw,
            cls.run_ppworkchain_e1_saw,
            cls.retrieve_potentials,
            cls.compute_epsilon_profile,
        )
        spec.dynamic_output()

    def initializing_e0(self):
        """
        Initializing inputs for the first calculation with a field amplitude of 0.0 a.u.
        """

        self.ctx.parameters_pp = ParameterData(
            dict={'INPUTPP': {
                'plot_num': 11,
            }})
        parameters = self.inputs.parameters.get_dict()
        if str(self.inputs.epsilon_type) == 'high-frequency':
            parameters['CONTROL']['calculation'] = 'scf'
        elif str(self.inputs.epsilon_type) == 'low-frequency':
            parameters['CONTROL']['calculation'] = 'relax'
        parameters['CONTROL']['tefield'] = True
        parameters['CONTROL']['dipfield'] = True
        #parameters['ELECTRON']['conv_thr'] = 1e-10
        parameters['SYSTEM']['edir'] = int(self.inputs.edir)
        parameters['SYSTEM']['emaxpos'] = float(self.inputs.emaxpos)
        parameters['SYSTEM']['eopreg'] = float(self.inputs.eopreg)
        parameters['SYSTEM']['eamp'] = 0.0  #int(self.inputs.eamp)
        self.ctx.parameters = ParameterData(dict=parameters)

        self.ctx.inputs_e0 = {
            'structure': self.inputs.structure,
            'code_pw': self.inputs.code_pw,
            'pseudo_family': self.inputs.pseudo_family,
            'kpoints': self.inputs.kpoints,
            'parameters': self.ctx.parameters,
            'parameters_pp': self.ctx.parameters_pp,
            'settings': self.inputs.settings,
            'options': self.inputs.options,
            'code_pp': self.inputs.code_pp,
            'pw_calc': Bool(True),
            'B_atom': self.inputs.B_atom,
            'magnetic_phase': self.inputs.magnetic_phase,
        }

    def run_ppworkchain_e0(self):
        """
        Running PpWorkChain to compute the electrostatic potential (V0) for tha case in which the amplitude
        if the applied field is 0 a.u.
        """

        running = submit(PpWorkChain, **self.ctx.inputs_e0)
        self.report(
            'Launching PpWorkChain for a FEF calculation with amplitude 0.0 a.u.. pk value {}'
            .format(running.pid))
        return ToContext(ppcalc_e0=running)

    def initializing_e1(self):
        """
        Initializing inputs for the  calculation with a field amplitude different from 0.0 a.u.
        """
        parameters_pp = ParameterData(dict={'INPUTPP': {
            'plot_num': 11,
        }})
        parameters = self.inputs.parameters.get_dict()

        if str(self.inputs.epsilon_type) == 'high-frequency':
            parameters['CONTROL']['calculation'] = 'scf'
        elif str(self.inputs.epsilon_type) == 'low-frequency':
            parameters['CONTROL']['calculation'] = 'relax'
        parameters['CONTROL']['tefield'] = True
        parameters['CONTROL']['dipfield'] = True
        #parameters['ELECTRON']['conv_thr'] = 1e-10
        parameters['SYSTEM']['edir'] = int(self.inputs.edir)
        parameters['SYSTEM']['emaxpos'] = float(self.inputs.emaxpos)
        parameters['SYSTEM']['eopreg'] = float(self.inputs.eopreg)
        parameters['SYSTEM']['eamp'] = float(self.inputs.eamp)
        self.ctx.parameters_e1 = ParameterData(dict=parameters)

        self.ctx.inputs_e1 = {
            'structure': self.inputs.structure,
            'code_pw': self.inputs.code_pw,
            'pseudo_family': self.inputs.pseudo_family,
            'kpoints': self.inputs.kpoints,
            'parameters': self.ctx.parameters_e1,
            'parameters_pp': parameters_pp,
            'settings': self.inputs.settings,
            'options': self.inputs.options,
            'code_pp': self.inputs.code_pp,
            'pw_calc': Bool(True),
            'B_atom': self.inputs.B_atom,
            'magnetic_phase': self.inputs.magnetic_phase,
        }

    def run_ppworkchain_e1(self):
        """
        Running PpWorkChain to compute the electrostatic potential (V1) for the case in which the amplitude
        if the applied field is different from 0 a.u.
        """
        running = submit(PpWorkChain, **self.ctx.inputs_e1)
        self.report(
            'Launching PpWorkChain for a FEF calculation with amplitude {} a.u.. pk value {}'
            .format(self.inputs.eamp, running.pid))
        return ToContext(ppcalc_e1=running)

    def initializing_e1_saw(self):
        """
        Initializing inputs for the calculation of the sawtooth potential.
        The PpWorkChain is  run using the wavefunction and all the necessary information  taken from 
        the PwCalculation performed in the previous step.
        """

        parameters_pp = ParameterData(dict={'INPUTPP': {
            'plot_num': 12,
        }})
        parent_folder = self.ctx.ppcalc_e1.out.remote_folder

        self.ctx.inputs_e1_saw = {
            'structure': self.inputs.structure,
            'parameters_pp': parameters_pp,
            'settings': self.inputs.settings,
            'options': self.inputs.options,
            'code_pp': self.inputs.code_pp,
            'pw_calc': Bool(False),
            'remote_folder': parent_folder,
        }

    def run_ppworkchain_e1_saw(self):
        """
        Running the PpWorkChain to calculate the sawtooth potential (V_saw)
        """
        running = submit(PpWorkChain, **self.ctx.inputs_e1_saw)
        self.report(
            'Launching PpWorkChain to compute the sawtooth potential. pk value {}'
            .format(running.pid))
        return ToContext(ppcalc_e1_saw=running)

    def retrieve_potentials(self):
        """
        Retreiving the 3D-FFT grid for each of the three potentials and storing the grids in the DB
        """
        self.ctx.grid_e0 = read_grid(self.ctx.ppcalc_e0.out.retrieved)
        self.ctx.grid_e1 = read_grid(self.ctx.ppcalc_e1.out.retrieved)
        self.ctx.grid_e1_saw = read_grid(self.ctx.ppcalc_e1_saw.out.retrieved)
        self.out('fft_grid_V0', self.ctx.grid_e0['fft_grid'])
        self.out('fft_grid_V1', self.ctx.grid_e1['fft_grid'])
        self.out('fft_grid_V_saw', self.ctx.grid_e1_saw['fft_grid'])

    def compute_epsilon_profile(self):
        """
        Computing the dielectric constant profile epsilon=(dV_saw/dz)/(d(V1-V0)/dz)
        """
        if int(self.inputs.edir) == 1:
            axis = 'x'
        elif int(self.inputs.edir) == 2:
            axis = 'y'
        elif int(self.inputs.edir) == 3:
            axis = 'z'

        epsilon = dielectric_profile_along_axis(
            self.inputs.structure, self.inputs.structure, self.ctx.grid_e0,
            self.ctx.grid_e1, self.inputs.structure, self.ctx.grid_e1_saw,
            axis, float(self.inputs.sigma))
        self.out('epsilon', epsilon)
        self.report("SlabEpsilonSawtoothWorkChain completed succesfully")


#@workfunction
def dielectric_profile_along_axis(structure_a,
                                  structure_b,
                                  potential_a,
                                  potential_b,
                                  structure_saw,
                                  sawtooth,
                                  axis,
                                  sigma=-100.):
    """
    Compute dielectric constant profile along one axis.
    Can be used for calculation on slabs
    :param structure_a/b: StructureData object containing the (optimized) structure in the absence or in presence
    of  uniform electric field, respectively
    :param potential_a/b: 3D-FFT grid  obtained from the read_grid function in both cases
    :param E_0: Float with the intensity of the applied external field in eV
    :param axis: Str with possible values "x", "y", "z" to indicate the direction along which epsilon is computed
    :param sigma: standard deviation of the gaussian kernel in angstrom
    :return epsilon: ArrayData object with the dielectric constant profile. epislon will be the low frequency 
    (epsilon_zero) if  atoms are optimized, or the high frequency epsilon_inf if the position are fixed
    References: Sundaraman and Ping JCP 146 104109 (2017), Pham  PRB 84, 045308 (2011)
    
    epsilon = (dV^saw/dz)/(dDektaV^scf/dz)
    
    NOTES:
    - It works correctly only if structure_a and structure_b have the same lenght along
    the direction along which the difference is computed (it is ok we only do relax not vc-relax)
    - If sigma it is not specified a value that is equal to half of the largest distance between 
    neighboring planes will be used. See (Pham et all.). However this can be very large.
    - PLEASE DO NOT PUT THE STRUCTURE AT THE EDGES OF THE SLAB!! 
    """
    from scipy.ndimage.filters import gaussian_filter
    from math import sqrt
    from mpmath import nint
    from scipy import signal

    #Computing the planar avareged electrostatic potential V along the axis
    V_a = planar_average(potential_a, structure_a, str(axis), npt=400)
    V_b = planar_average(potential_b, structure_b, str(axis), npt=400)
    E = planar_average(sawtooth, structure_saw, str(axis), npt=400)

    #Computing Delta V
    Vdiff = V_a['average'] - V_b['average']
    ax = V_a['ax']

    #Identifying position of the slab/vacuum interface using the coordinates of the outermost atoms
    cell = structure_a.cell
    coords = []
    if axis == 'x':
        lenght = (sqrt(cell[0][0]**2 + cell[0][1]**2 + cell[0][2]**2))
        for site in structure_a.sites:
            coords.append(
                float(site.position[0] -
                      nint(site.position[0] / lenght) * lenght))

    if axis == 'y':
        lenght = (sqrt(cell[1][0]**2 + cell[1][1]**2 + cell[1][2]**2))
        for site in structure_a.sites:
            coords.append(
                flaot(site.position[1] -
                      nint(site.position[1] / lenght) * lenght))
    if axis == 'z':
        lenght = (sqrt(cell[2][0]**2 + cell[2][1]**2 + cell[2][2]**2))
        for site in structure_a.sites:
            coords.append(
                float(site.position[2] -
                      nint(site.position[2] / lenght) * lenght))

    coords = sorted(coords, key=int)

    vacuum_max = None
    vacuum_min = None
    plane_dists = []

    for i in range(len(coords) - 1):
        plane_dists.append(abs(coords[i] - coords[i + 1]))
        print(i, coords[i], coords[i + 1], abs(coords[i] - coords[i + 1]))
        if abs(coords[i] - coords[i + 1]) > 5.:
            vacuum_max = coords[i + 1]
            vacuum_min = coords[i]

    if vacuum_max == None or v_min == None:
        vacuum_max = max(coords)
        vacuum_min = min(coords) + lenght

    #print vacuum_max, vacuum_min

    vacuum_center = (vacuum_max + vacuum_min) * 0.5
    dist = abs(vacuum_max - vacuum_center)
    smooth_percent = dist * 0.85

    lim1 = vacuum_center - smooth_percent
    lim2 = vacuum_center + smooth_percent
    #print "lim1, lim2", lim1, lim2

    #Computing Delta_V derivative
    der_Vdiff = differentiator(ax, Vdiff)

    #Computing Sawtooth potential derivative
    der_E = differentiator(E['ax'], E['average'])

    #Calculating epsilon
    eps = der_E / der_Vdiff

    #Imposing that in the vacuum region epsilon is equal to 1:
    for i in range(len(ax)):
        if ax[i] > lim1 and ax[i] < lim2:
            eps[i] = 1.0

    #Smoothing epsilon with Gaussian kernel
    ncol = len(eps)
    if sigma == -100.:
        sigma = max(plane_dists) * 0.5
    sigma *= ncol / lenght

    #print "SIGMA", sigma
    kernel = signal.gaussian(ncol, sigma)
    kernel /= kernel.sum()
    epsilon = signal.fftconvolve(eps, kernel, mode='same')

    #Storing the dielectric profile into an 2D ArrayData object with the axis and dielectric constant values
    profile = np.vstack((ax, epsilon)).T
    slabeps = ArrayData()
    slabeps.set_array('epsilon', profile)

    #Storing the profile also into a file
    plt.plot(ax, epsilon)
    plt.xlabel(str(axis) + ur' (\u00c5)', fontsize=13)
    plt.ylabel(r'$\epsilon$(' + str(axis) + ')', fontsize=12)
    plt.savefig(str(structure_a.pk) + '_epsilon.pdf')
    plt.show()
    return slabeps
