============
Installation
============

Install from anaconda (Coming Soon)
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

2. Set up and activate environment:
::

    $ mamba env create -f environment.yml
    $ mamba activate msibi
    $ python -m pip install .

.. note::

    To install a GPU compatible version of HOOMD-blue in your msibi environment, you need to manually set the CUDA version **before installing msibi**.
    This is to ensure that the HOOMD build pulled from conda-forge is compatible with your CUDA version.
    To set the CUDA version, run the following command before installing msibi::

        $ export CONDA_OVERRIDE_CUDA="[YOUR_CUDA_VERSION]"

    Please see the `HOOMD-blue installation instructions <https://hoomd-blue.readthedocs.io/en/stable/installation.html>`_ for more information.
