Welcome to AiiDA-Defects
++++++++++++++++++++++++

AiiDA-Defects is a plugin for the `AiiDA <http://www.aiida.net/>`_ computational 
materials science framework, and provides tools and automated workflows for the 
study of defects in materials.

The package is available for download from `GitHub <http://github.com/aiida-defects>`_.

If you use AiiDA-Defects in your work, please cite:

    *paper reference (doi)*

Please also remember to cite the `AiiDA paper <https://doi.org/10.1016/j.commatsci.2015.09.013>`_.


Quick Setup
===========

Install this package by running the following in your shell:

    .. code-block:: bash

        $ pip install .[docs]

This will install all of the prerequisites automatically (including for the optional docs) 
in your environment, including AiiDA core, if it not already installed. 
Ideally however, you should install AiiDA-Defects after installing and setting 
up AiiDA core.

To build the local docs, run:

    .. code-block:: bash

        $ cd docs/
        $ make html

Note: You will need to have ``make`` installed on your operating system.


Acknowledgements
================
This work is funded by...