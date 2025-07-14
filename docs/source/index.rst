MSIBI: Multi-state Iterative Boltzmann Inversion
=============================================================================================

This is a python package that implements the MSIBI coarse-graining method for molecular dynamics.

This implementation provies an intuitive python API for running iterative Boltzmann inversion (IBI) for multiple states (MSIBI) or a single state.
It is designed to easily enable stringing together multiple optimization runs where after one coarse-grained interaction is learned (i.e., bond stretching) it can be included and held constant during the the next interaction optimization (e.g., bond bending and non-bonded pairs).
Iterative coarse-grain simulations use the HOOMD-Blue simulation engine under-the-hood; however it is not required that the target (i.e., atomistic) simulations are performed with HOOMD-Blue, and the learned coarse-grained potentials are in a table potential format, which are usable with other molecular dynamics engines (e.g., LAMMPS, GROMACS).

Quick start
===========
.. toctree::
    installation


Resources
=========

- `GitHub Repository <https://github.com/mosdef-hub/msibi>`_: Source code and issue tracker.

- `MSIBI paper <https://doi.org/10.1063/1.4880555>`_: Explanation of the MSIBI method.

- `HOOMD-Blue <https://hoomd-blue.readthedocs.io/en/latest/>`_: Python package used to perform molecular dynamics simulations on CPUs and GPUs.


.. toctree::
   :maxdepth: 1
   :caption: Python API

   optimize
   forces
   state


Citation
========
If you use the ``msibi`` python package in your research, please cite the following paper:

.. code-block:: bibtex

   @article{Moore2014,
         author = "Moore, Timothy C. and Iacovella, Christopher R. and McCabe, Clare",
         title = "Derivation of coarse-grained potentials via multistate iterative Boltzmann inversion",
         journal = "The Journal of Chemical Physics",
         year = "2014",
         volume = "140",
         number = "22",
         doi = "http://dx.doi.org/10.1063/1.4880555"
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
