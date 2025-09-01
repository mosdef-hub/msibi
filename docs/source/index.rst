MSIBI: Multi-state Iterative Boltzmann Inversion
=============================================================================================

**MSIBI** is a Python package that implements the Multi-state Iterative Boltzmann Inversion (MS-IBI) method for coarse-graining in molecular dynamics.

This implementation provides an intuitive Python API for running iterative Boltzmann inversion (IBI) across multiple states (MS-IBI) or a single state. It is designed to make it easy to chain together multiple optimization runs—for example, you can first optimize a coarse-grained bond-stretching interaction, then hold it fixed while optimizing other interactions like bond bending or non-bonded pair potentials.

MSIBI uses the HOOMD-blue simulation engine under the hood for coarse-grained simulations; however, it does not require that the target (atomistic) simulations are also performed with HOOMD-blue. The resulting coarse-grained potentials are exported in a tabulated format compatible with other simulation engines such as LAMMPS and GROMACS.


Quick start
===========
.. toctree::
    :maxdepth: 1

    installation
    examples
    tutorials/01-Introduction


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
