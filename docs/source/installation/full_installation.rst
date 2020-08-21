.. full_installation:

Detailed Setup
==============

1. Prerequisites
----------------

There are a couple of software packages on which AiiDA-Defects relies, in 
addition to AiiDA core and its dependencies.

The minimum requirements are:

- aiida_core >=0.12.0, <1.0.0,
- aiida-quantumespresso >= 2.1.0, <3.0.0
- pymatgen

These will be installed automatically by ``pip`` if not available in your
environment, but we recommend that you install and setup AiiDA core first, 
before setting up AiiDA-Defects. Instructions for installing AiiDA core can 
be found in the 
`AiiDA documentation <https://www.aiida-core.readthedocs.io/en/stable/>`_ .

To build the documentation locally, the following are also required:

- sphinx 
- docutils
- sphinx_rtd_theme

Again, these will be automatically installed by ``pip``.


2a. Installing AiiDA-Defects from PyPi
--------------------------------------

To download and install the latest stable release of AiiDA-Defects, 
simply run from a shell:

    .. code-block:: bash

        $ pip install aiida-defects

This will fetch the latest release from the PyPi repository, install of the 
dependencies, before finally installing AiiDA-Defects.

If you wish to have the documentation available locally, instead you 
should run:

    .. code-block:: bash

        $ pip install aiida-defects[docs]

This will do all the of the above, as well as installing the additional 
packages needed for building the documentation.


2b. Installing AiiDA-Defects from Git
-------------------------------------

Alternatively, to download and install the latest development version of 
AiiDA-Defects, first clone the Git repository from a shell by running:

    .. code-block:: bash

        $ git clone https://github.com/aiidateam/aiida_core.git

Note: You will need to have ``git`` installed on your operating system.

Navigate to the directory that is created and install via ``pip``:

    .. code-block:: bash

        $ cd aiida-defects
        $ pip install -e .

This will install the cloned version of AiiDA-Defects, as well as fetching
any missing dependencies. 

.. tip::
    Installing via ``pip`` using the ``-e`` option allows the source code to
    be updated without reinstalling. This allow you to easily get and apply 
    the latest updates from the development version via ``git pull`` .


.. warning::
    The development version from ``git`` is not the latest release and may 
    include undiscovered bugs or unfinished experimental features, and so 
    is not recommended for production runs.


If you wish to have the documentation available locally, instead you 
should run:

    .. code-block:: bash

        $ pip install -e .[docs]

This will do all the of the above, as well as installing the additional 
packages needed for building the documentation.


3. Building the Documentation
-----------------------------

If you chose to install the optional prerequisites for the documentation, 
you can build the local documentation. 

Navigate to the ``docs`` directory and run ``make``:

    .. code-block:: bash

        $ cd docs/
        $ make html

Note: You will need to have ``make`` installed on your operating system.

This will build the HTML docs in the ``build/html/`` directory.
The docs can then viewed in any web browser by navigating to, for example:

    .. code-block:: bash

        file:///home/conrad/aiida-defects/docs/build/html/index.html

The exact path depends on your installation location and the exact prefix 
depends on your web browser. Here we use ``file://`` which is standard for 
Firefox, Opera, Chrome and Safari.

If ``latex`` is available on the system, one can also generate a PDF version
of the documentation by running:

    .. code-block:: bash

        $ cd docs/
        $ make latex
        $ cd build/latex
        $ pdflatex aiida-defects.tex

If compilation by ``pdflatex`` is successful, ``aiida-defects.pdf`` will 
be placed in ``docs/build/latex/``.