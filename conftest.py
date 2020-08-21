"""
For pytest 
initialise a text database and profile
"""
from __future__ import absolute_import
import tempfile
import shutil
import pytest

from aiida.manage.fixtures import fixture_manager


@pytest.fixture(scope='session', autouse=True)
def aiida_profile():
    """Set up a test profile for the duration of the tests"""
    with fixture_manager() as fixture_mgr:
        yield fixture_mgr


@pytest.fixture(scope='function', autouse=True)
def clear_database(aiida_profile):
    """Clear the database after each test"""
    yield
    aiida_profile.reset_db()


@pytest.fixture(scope='function')
def new_workdir():
    """Get a temporary folder to use as the computer's work directory."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)

@pytest.fixture(scope='class')
def test_structures():
    """Get a library of structure data objects for use in tests"""
    import pkg_resources
    import glob
    import pymatgen
    from aiida.orm import StructureData

    structures_dict = {}

    data_dir = pkg_resources.resource_filename('aiida_defects', '/tests/test_data/')

    # Look for cif files in the data dir, convert them to StructureData objects and 
    # store in the dictionary with the filename as the key
    for file_path in glob.glob(data_dir+'*.cif'):
        structure_mg= pymatgen.Structure.from_file(file_path)
        structure_sd = StructureData(pymatgen=structure_mg)
        label = file_path.split('/')[-1].rstrip('.cif')
        structures_dict[label] = structure_sd
    
    return structures_dict   

# @pytest.fixture(scope='function')
# def aiida_localhost_computer(new_workdir):
#     """Get an AiiDA computer for localhost.

#     :return: The computer node
#     :rtype: :py:class:`aiida.orm.Computer`
#     """
#     from aiida_defects.helpers import get_computer

#     computer = get_computer(workdir=new_workdir)

#     return computer


# @pytest.fixture(scope='function')
# def aiida_code(aiida_localhost_computer):
#     """Get an AiiDA code.

#     :return: The code node
#     :rtype: :py:class:`aiida.orm.Code`
#     """
#     from aiida_defects.helpers import get_code

#     code = get_code(entry_point='aiida_defects', computer=aiida_localhost_computer)

#     return code