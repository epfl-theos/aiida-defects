# -*- coding: utf-8 -*-
########################################################################################
# Copyright (c), The AiiDA-Defects authors. All rights reserved.                       #
#                                                                                      #
# AiiDA-Defects is hosted on GitHub at https://github.com/epfl-theos/aiida-defects     #
# For further information on the license, see the LICENSE.txt file                     #
########################################################################################
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pkg_resources
import pytest
import six
from six.moves import range


class TestDefects(object):
    """
    Tests for tools.defects module
    """

    def test_defect_creator_vacancy(self, aiida_profile, test_structures):
        """
        Test defect_creator functionality for vacancies
        Expected result - Creates correct vacancy structure
        """
        import aiida_defects.tools.defects
        from aiida.orm.data.parameter import ParameterData
        from aiida.orm.data.base import Bool, List

        unitcell_structure = test_structures['halite_unitcell']

        vacancies = List()
        vacancies._set_list(['Cl'])
        substitutions = ParameterData(dict={})
        supercell_scale = List()
        supercell_scale._set_list([2, 2, 2])
        cluster = Bool(False)

        new_structures = aiida_defects.tools.defects.defect_creator(
            unitcell_structure, vacancies, substitutions, supercell_scale,
            cluster)

        # Check that the structures created are valid
        # Supercell cell vectors should be twice the unit cell
        host_unitcell = np.asarray(unitcell_structure.cell)
        host_supercell = np.asarray(new_structures['vacancy_0'].cell)
        np.testing.assert_allclose(
            host_unitcell * 2, host_supercell, atol=1.e-9)

        # Check that a vacancy has been created
        host_site_list = new_structures['vacancy_0'].sites
        defect_site_list = new_structures['vacancy_1'].sites

        # Find the defect site
        vacancy_sites = []
        for host_site in host_site_list:
            for defect_site in defect_site_list:
                if host_site.get_raw() == defect_site.get_raw():
                    break
            else:
                vacancy_sites.append(host_site)

        reference = {
            'position': (2.72665, 2.72665, 2.72665),
            'kind_name': 'Cl'
        }
        result = vacancy_sites[0].get_raw()

        # There should only be one vacancy
        assert len(vacancy_sites) == 1
        # The position of the defect should be arbitrarily close to where it is expected
        np.testing.assert_allclose(
            np.asarray(result.pop('position')),
            np.asarray(reference.pop('position')),
            atol=1.e-9)
        # Iterate over remaining keys and check their values
        for key, value in six.iteritems(reference):
            assert value == result.pop(key)

    def test_defect_creator_substitution(self, aiida_profile, test_structures):
        """
        Test defect_creator functionality for substitution
        Expected result - Creates correct substitution structure
        """
        import aiida_defects.tools.defects
        from aiida.orm.data.parameter import ParameterData
        from aiida.orm.data.base import Bool, List

        unitcell_structure = test_structures['halite_unitcell']
        vacancies = List()
        vacancies._set_list([])  # Workaround for issue #1171
        substitutions = ParameterData(dict={'Na': ['K']})
        supercell_scale = List()
        supercell_scale._set_list([2, 2, 2])
        cluster = Bool(False)

        new_structures = aiida_defects.tools.defects.defect_creator(
            unitcell_structure, vacancies, substitutions, supercell_scale,
            cluster)

        # Check that the structures created are valid
        # Supercell cell vectors should be twice the unit cell
        host_unitcell = np.asarray(unitcell_structure.cell)
        host_supercell = np.asarray(new_structures['substitution_0'].cell)
        np.testing.assert_allclose(
            host_unitcell * 2, host_supercell, atol=1.e-9)

        # Check that a substitution has been created
        host_site_list = new_structures['substitution_0'].sites
        defect_site_list = new_structures['substitution_1'].sites

        # Find the defect site
        substitution_sites = []
        for host_site in host_site_list:
            for defect_site in defect_site_list:
                if host_site.get_raw() == defect_site.get_raw():
                    break
            else:
                substitution_sites.append(defect_site)

        reference = {'position': (0., 0., 0.), 'kind_name': 'K'}
        result = substitution_sites[0].get_raw()

        # There should only be one substitution
        assert len(substitution_sites) == 1
        # The position of the defect should be arbitrarily close to where it is expected
        np.testing.assert_allclose(
            np.asarray(result.pop('position')),
            np.asarray(reference.pop('position')),
            atol=1.e-9)
        # Iterate over remaining keys and check their values
        for key, value in six.iteritems(reference):
            assert value == result.pop(key)

    @pytest.mark.skip(
        reason=
        "Investigation needed - cluster creator doesn't return the expected clusters"
    )
    def test_defect_creator_cluster(self, aiida_profile, test_structures):
        """
        Test defect_creator functionality for clusters
        Expected result - Creates correct cluster structures
        """
        import aiida_defects.tools.defects
        from aiida.orm.data.parameter import ParameterData
        from aiida.orm.data.base import Bool, List

        unitcell_structure = test_structures['halite_unitcell']
        vacancies = List()
        vacancies._set_list(['Cl'])
        substitutions = ParameterData(dict={'Na': ['K']})
        supercell_scale = List()
        supercell_scale._set_list([2, 2, 2])
        cluster = Bool(True)

        new_structures = aiida_defects.tools.defects.defect_creator(
            unitcell_structure, vacancies, substitutions, supercell_scale,
            cluster)

        # # Export the structures as cif files for debugging
        # import pymatgen.io.aiida
        # aiida_structure_adaptor = pymatgen.io.aiida.AiidaStructureAdaptor()
        # import pymatgen.io.cif

        # for label,struct in new_structures.iteritems():
        #     structure_pymat = aiida_structure_adaptor.get_structure(struct)
        #     print('\n\n\n')
        #     print(struct.sites)
        #     print('\n\n\n')
        #     assert len(structure_pymat.sites) == len(struct.sites)
        #     pymatgen.io.cif.CifWriter(structure_pymat).write_file('./new_cifs/'+label+'.cif')

        # Check that the structures created are valid
        # Supercell cell vectors should be twice the unit cell
        host_unitcell = np.asarray(unitcell_structure.cell)
        host_supercell = np.asarray(new_structures['cluster_0'].cell)
        np.testing.assert_allclose(
            host_unitcell * 2, host_supercell, atol=1.e-9)

        # For a 2x2x2 supercell, there should be 4 possible clusters (and one host structure)
        assert len(new_structures) == 5

        # Check that the defects are in the right sites
        # Like in the tests for vacancies, there could be multiple sites where these could be
        # inserted, due to symmetry, but here we check for a particular site in
        # order to detect code regressions

        host_site_list = new_structures['cluster_0'].sites

        references = {
            'cluster_1': [
                {
                    'position': (0., 0., 0.),
                    'kind_name': 'K'
                },
            ],
        }

        # Find the defect sites
        for index in range(1, 5):

            cluster_label = 'cluster_' + str(index)
            defect_site_list = new_structures[cluster_label].sites

            cluster_sites = []
            for host_site in host_site_list:
                for defect_site in defect_site_list:
                    if host_site.get_raw() == defect_site.get_raw():
                        break
                else:
                    cluster_sites.append(defect_site)

            # Only two defects in the cluster
            assert len(cluster_sites) == 2

            # The position of the defect sites should be arbitrarily close to
            # where they are expected. The sites aren't sorted so we need to try
            # and match each defect site to a reference site
            failed_sites = []
            for defect_site in cluster_sites:
                defect_site = defect_site.get_raw()
                for ref_site in references[cluster_label]:
                    # Check positions
                    is_match_pos = np.allclose(
                        np.asarray(defect_site.pop('position')),
                        np.asarray(ref_site['position']),
                        atol=1.e-9)
                    # Check labels
                    if defect_site.pop('kind_name') == ref_site['kind_name']:
                        is_match_elem = True
                    else:
                        is_match_elem = False
                    # No keys left in the defect site definition
                    assert len(
                        defect_site
                    ) == 0, "Unexpected key found in cluster site definition"
                    # Matching site found?
                    if is_match_pos and is_match_elem:
                        break
                else:
                    # Site not among reference site list
                    failed_sites.append(defect_site)

            assert len(failed_sites) == 0, "Found unexpected defect sites"

    def defect_creator_by_index(structure, find_defect_index_output):
        """
        Test defect_creator functionality for vacancies
        Expected result - Creates correct vacancy structure
        """
        import aiida_defects.tools.defects
        from aiida.orm.data.parameter import ParameterData
        from aiida.orm.data.base import Bool, List

        unitcell_structure = test_structures['halite_unitcell']

        vacancies = List()
        vacancies._set_list(['Cl'])
        substitutions = ParameterData(dict={})
        supercell_scale = List()
        supercell_scale._set_list([2, 2, 2])
        cluster = Bool(False)

        new_structures = aiida_defects.tools.defects.defect_creator(
            unitcell_structure, vacancies, substitutions, supercell_scale,
            cluster)

        # Check that the structures created are valid
        # Supercell cell vectors should be twice the unit cell
        host_unitcell = np.asarray(unitcell_structure.cell)
        host_supercell = np.asarray(new_structures['vacancy_0'].cell)
        np.testing.assert_allclose(
            host_unitcell * 2, host_supercell, atol=1.e-9)

        # Check that a vacancy has been created
        host_site_list = new_structures['vacancy_0'].sites
        defect_site_list = new_structures['vacancy_1'].sites

        # Find the defect site
        vacancy_sites = []
        for host_site in host_site_list:
            for defect_site in defect_site_list:
                if host_site.get_raw() == defect_site.get_raw():
                    break
            else:
                vacancy_sites.append(host_site)

        reference = {
            'position': (2.72665, 2.72665, 2.72665),
            'kind_name': 'Cl'
        }
        result = vacancy_sites[0].get_raw()

        # There should only be one vacancy
        assert len(vacancy_sites) == 1
        # The position of the defect should be arbitrarily close to where it is expected
        np.testing.assert_allclose(
            np.asarray(result.pop('position')),
            np.asarray(reference.pop('position')),
            atol=1.e-9)
        # Iterate over remaining keys and check their values
        for key, value in six.iteritems(reference):
            assert value == result.pop(key)

    def test_distance_from_defect(self, aiida_profile, test_structures):
        """
        Test distance_from_defect function.
        Expected result - array of distances computed matches a reference array
        """

        import aiida_defects.tools.defects

        results = aiida_defects.tools.defects.distance_from_defect(
            test_structures['lton_bulk'], np.array([0., 0., 0.]))

        distances_calculated = []
        for item in results:
            distances_calculated.append(item[1])
        distances_calculated = np.asarray(distances_calculated)

        distances_reference = np.array([
            0.0, 3.421396265623875, 2.7682664798028385, 2.80611880625625,
            2.80611880625625
        ])

        # Compare the calculated values with the reference values to ensure they are arbitrarily close
        # Fails if any one value is not within
        np.testing.assert_allclose(
            distances_calculated, distances_reference, atol=1.e-9)

    def test_distance_from_defect_aiida(self, aiida_profile, test_structures):
        """
        Test distance_from_defect_aiida function.
        Expected result - array of distances computed matches a reference array
        """

        import aiida_defects.tools.defects

        results = aiida_defects.tools.defects.distance_from_defect_aiida(
            test_structures['lton_bulk'], np.array([0., 0., 0.]))

        distances_calculated = []
        for item in results:
            distances_calculated.append(item[1])
        distances_calculated = np.asarray(distances_calculated)

        distances_reference = np.array([
            0.0, 3.421396265623875, 2.7682664798028385, 2.80611880625625,
            2.80611880625625
        ])

        # Compare the calculated values with the reference values to ensure they are arbitrarily close
        # Fails if any one value is not within
        np.testing.assert_allclose(
            distances_calculated, distances_reference, atol=1.e-9)

    def test_distance_from_defect_pymatgen(self, aiida_profile,
                                           test_structures):
        """
        Test distance_from_defect_pymatgen function.
        Expected result - array of distances computed matches a reference array
        """

        import aiida_defects.tools.defects

        results = aiida_defects.tools.defects.distance_from_defect_pymatgen(
            test_structures['lton_bulk'], np.array([0., 0., 0.]))

        distances_calculated = []
        for item in results:
            distances_calculated.append(item[1])
        distances_calculated = np.asarray(distances_calculated)

        distances_reference = np.array([
            0.0, 3.421396265623875, 2.7682664798028385, 2.80611880625625,
            2.80611880625625
        ])

        # Compare the calculated values with the reference values to ensure they are arbitrarily close
        # Fails if any one value is not within
        np.testing.assert_allclose(
            distances_calculated, distances_reference, atol=1.e-9)

    @pytest.mark.skip(
        reason=
        "Cluster returns a list instead of a numpy array and breaks the comparison"
    )
    def test_explore_defect(self, aiida_profile, test_structures):
        """
        Test explore_defect function.
        Expected result - correct dicts returned describing known defect
        """
        import aiida_defects.tools.defects

        host_structure = test_structures['halite_bulk']

        # Case 1 - Vacancy
        defect_structure = test_structures['halite_bulk_v_cl']
        result = aiida_defects.tools.defects.explore_defect(
            host_structure, defect_structure, 'vacancy')

        reference = {
            'defect_name': 'V_Cl',
            'defect_position': np.array([7.929389, 7.929389, 7.929389]),
            'atom_type': 'Cl'
        }

        # Check that the defect positions are arbitrarily close
        np.testing.assert_allclose(
            result.pop('defect_position'),
            reference.pop('defect_position'),
            atol=1.e-9)
        # Iterate over remaining keys and check their values
        for key, value in six.iteritems(reference):
            assert value == result.pop(key)

        # There should be no keys left
        assert len(result) == 0

        # Case 2 - Substitution
        defect_structure = test_structures['halite_bulk_sub_k']
        result = aiida_defects.tools.defects.explore_defect(
            host_structure, defect_structure, 'substitution')

        reference = {
            'defect_name': 'Na_K',
            'defect_position': np.array([4.940871, 7.929389, 7.929389]),
            'atom_type': 'K'
        }

        # Check that the defect positions are arbitrarily close
        np.testing.assert_allclose(
            result.pop('defect_position'),
            reference.pop('defect_position'),
            atol=1.e-9)
        # Iterate over remaining keys and check their values
        for key, value in six.iteritems(reference):
            assert value == result.pop(key)

        # There should be no keys left
        assert len(result) == 0

        # Case 3 - Cluster
        defect_structure = test_structures['halite_bulk_v_cl_sub_k']
        result = aiida_defects.tools.defects.explore_defect(
            host_structure, defect_structure, 'cluster')

        print(result)

        reference = {
            'defect_name_s_0': 'Na_K',
            'defect_name_v_0': 'V_Cl',
            'atom_type_s_0': 'K',
            'atom_type_v_0': 'Cl',
            'defect_position_s_0': np.array([0.453, 0.727, 0.727]),
            'defect_position_v_0': np.array([7.929389, 7.929389, 7.929389])
        }

        # Check that the defect positions are arbitrarily close
        np.testing.assert_allclose(
            result.pop('defect_position_s_0'),
            reference.pop('defect_position_s_0'),
            atol=1.e-9)
        np.testing.assert_allclose(
            result.pop('defect_position_v_0'),
            reference.pop('defect_position_v_0'),
            atol=1.e-9)

        # Iterate over remaining keys and check their values
        for key, value in six.iteritems(reference):
            assert value == result.pop(key)

        # There should be no keys left
        assert len(result) == 0

        # Case 4 - Unknown
        defect_structure = test_structures['halite_bulk_v_cl_sub_k']
        result = aiida_defects.tools.defects.explore_defect(
            host_structure, defect_structure, 'unknown')

        print(result)

        reference = {
            'defect_name_s_0': 'Na_K',
            'defect_name_v_0': 'V_Cl',
            'atom_type_s_0': 'K',
            'atom_type_v_0': 'Cl',
            'defect_position_s_0': np.array([0.453, 0.727, 0.727]),
            'defect_position_v_0': np.array([7.929389, 7.929389, 7.929389])
        }

        # Check that the defect positions are arbitrarily close
        np.testing.assert_allclose(
            result.pop('defect_position_s_0'),
            reference.pop('defect_position_s_0'),
            atol=1.e-9)
        np.testing.assert_allclose(
            result.pop('defect_position_v_0'),
            reference.pop('defect_position_v_0'),
            atol=1.e-9)

        # Iterate over remaining keys and check their values
        for key, value in six.iteritems(reference):
            assert value == result.pop(key)

        # There should be no keys left
        assert len(result) == 0

    @pytest.mark.skip(
        reason=
        'Find index appears to be broken for clusters. Also need to make output keys more generic.'
    )
    def test_find_defect_index(self, aiida_profile, test_structures):
        """
        Test find_defect_index function
        Expected result - returns dictionary with correct defect information
        """
        import aiida_defects.tools.defects

        host_structure = test_structures['halite_bulk']

        # Case 1 - Point defect
        defect_structure = test_structures['halite_bulk_v_cl']
        structure_dict = {
            'vacancy_0': host_structure,
            'vacancy_1': defect_structure
        }
        result = aiida_defects.tools.defects.find_defect_index(structure_dict)

        reference = {
            'vacancy_1': {
                'defect_name': 'V_Cl',
                'index': 32,
                'atom_type': 'Cl',
                'defect_position': np.array([7.929389, 7.929389, 7.929389])
            }
        }

        for label, data in six.iteritems(result):
            np.testing.assert_allclose(
                result[label].pop('defect_position'),
                reference[label].pop('defect_position'),
                atol=1.e-9)
            for key, value in six.iteritems(data):
                assert reference[label][
                    key] == value, "The output dictionary key does not contain the expected value"

        # Case 2 - Cluster
        defect_structure = test_structures['halite_bulk_v_cl_sub_k']

        structure_dict = {
            'cluster_0': host_structure,
            'cluster_1': defect_structure
        }
        result = aiida_defects.tools.defects.find_defect_index(structure_dict)
        print(result)

        reference = {
            'cluster_1': {
                'defect_name_v_0': {
                    'index':
                    32,
                    'atom_type_v_0':
                    'Cl',
                    'defect_position_v_0':
                    np.array([7.929389, 7.929389, 7.929389]),
                },
                'defect_name_s_0': {}
            }
        }

        for cluster_label, cluster_data in six.iteritems(result):
            for defect_label, defect_data in six.iteritems(cluster_data):
                np.testing.assert_allclose(
                    result[cluster_label][defect_label].pop('defect_position'),
                    reference[cluster_label][defect_label].pop(
                        'defect_position'),
                    atol=1.e-9)
            for key, value in six.iteritems(data):
                assert reference[cluster_label][defect_label][
                    key] == value, "The output dictionary key does not contain the expected value"
