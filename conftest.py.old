"""
For pytest.
Initialise a text database and profile.
"""
from __future__ import absolute_import
import tempfile
import shutil
import pytest
import os
import glob
import pkg_resources

from aiida.utils.fixtures import fixture_manager

def get_backend_str():
    """ 
    Return database backend string.
    Reads from 'TEST_AIIDA_BACKEND' environment variable.
    Defaults to django backend.
    """
    from aiida.backends.profile import BACKEND_DJANGO, BACKEND_SQLA
    backend_env = os.environ.get('TEST_AIIDA_BACKEND')
    if not backend_env: 
        return BACKEND_DJANGO
    elif  backend_env in (BACKEND_DJANGO, BACKEND_SQLA):
        return backend_env

    raise ValueError("Unknown backend '{}' read from TEST_AIIDA_BACKEND environment variable".format(backend_env))


@pytest.fixture(scope='session')
def aiida_profile():
    """Setup a test profile for the duration of the tests."""
    with fixture_manager() as fixture_mgr:
        yield fixture_mgr


@pytest.fixture(scope='function')
def new_database(aiida_profile):
    """Clear the database after each test."""
    yield
    aiida_profile.reset_db()


@pytest.fixture(scope='function')
def new_workdir():
    """Get a new temporary folder to use as the computer's workdir."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture(scope='class')
def test_structures():
    """Get a library of structure data objects for use in tests"""
    import pymatgen.io.cif
    import pymatgen.io.aiida
    import aiida_defects.tools.defects

    aiida_structure_adaptor = pymatgen.io.aiida.AiidaStructureAdaptor()

    structures_dict = {}

    data_dir = pkg_resources.resource_filename('aiida_defects', '/tests/test_data/')

    # Look for cif files in the data dir, convert them to StructureData objects and 
    # store in the dictionary with the filename as the key
    for file_path in glob.glob(data_dir+'*.cif'):
        structure_mg= pymatgen.io.cif.Structure.from_file(file_path)
        structure_sd = aiida_structure_adaptor.get_structuredata(structure_mg)
        label = file_path.split('/')[-1].rstrip('.cif')
        structures_dict[label] = structure_sd
    
    return structures_dict   
