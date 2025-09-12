============
Installation
============

These examples use the `anaconda package manager <https://www.anaconda.com/download>`_.

Install from anaconda
---------------------------------------

msibi is available on `conda-forge <https://anaconda.org/conda-forge/msibi>`_
::

    $ mamba install -c conda-forge msibi


Install from source
---------------------------------------

1. Clone this repository:
::

    $ git clone git@github.com:mosdef-hub/msibi.git
    $ cd msibi

2. Install dependency packages and msibi from source:
::

    $ mamba env create -f environment.yml
    $ mamba activate msibi
    $ python -m pip install .

.. note::

    ``msibi`` utilizes HOOMD-Blue under-the-hood to run query simulations. This means that you are able to leverage HOOMD-Blue's GPU capability
    in your optimiation runs if your HOOMD-Blue installation is compatible with GPUs.
    To install a GPU compatible version of HOOMD-Blue in your msibi environment, you need to manually set the CUDA version **before installing msibi**.
    This is to ensure that the HOOMD build pulled from conda-forge is compatible with your CUDA version.
    To set the CUDA version, run the following command before installing msibi::

        $ export CONDA_OVERRIDE_CUDA="[YOUR_CUDA_VERSION]"

    Please see the `HOOMD-blue installation instructions <https://hoomd-blue.readthedocs.io/en/stable/installation.html>`_ for more information.
