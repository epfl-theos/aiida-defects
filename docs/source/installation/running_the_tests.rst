.. _running_the_tests:

Running the Tests 
=================

If desired, the unit tests that ship with AiiDA-Defects can be run.
This can be useful before a production run to ensure that everything is 
working as expected with your combination of system and installed packages.

To install the necessary dependencies, run from a shell:

    .. code-block:: bash

        $ pip install .[testing]

This will install all of the testing prerequisites automatically in your environment,
if not already installed.

The tests can then be run using:

    .. code-block:: bash

        $ pytest -v

In order for `pytest` to successfully discover all the tests, this should be run
from the top AiiDA-Defects directory. 

It is also possible to check the approximate extent to which the codebase is covered 
by testing, by instead running:
    
    .. code-block:: bash

        $ pytest -v --cov=aiida_defects

This will run the tests as well as printing a report detailing what fraction 
of the code base was visited during the testing.

.. note::
    Please let us know if any of the tests fail on your system.
    While we have endeavoured to test thoroughly, it is impossible to test  
    every possible combination of system and dependencies. 

