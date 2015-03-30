.. Multistate Iterative Boltzmann Inversion documentation master file, created by
   sphinx-quickstart on Mon Mar 30 14:37:56 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MultiState Iterative Boltzmann Inversion
----------------------------------------

A package to help you manage and run pair potential optimizations using the
multistate iterative Boltzmann inversion procedure.

Installation
------------
.. toctree::
    installation

Tutorials
---------
.. toctree::
   tutorials/tutorials

Citation
--------
Details of the underlying method and its validation can be found here |citation|

If you use this package please cite the above paper. The BibTeX reference is::

    @article{Moore2014,
          author = "Moore, Timothy C. and Iacovella, Christopher R. and McCabe, Clare",
          title = "Derivation of coarse-grained potentials via multistate iterative Boltzmann inversion",
          journal = "The Journal of Chemical Physics",
          year = "2014",
          volume = "140",
          number = "22",
          doi = "http://dx.doi.org/10.1063/1.4880555"
    }


API Reference
-------------
.. toctree::
    msibi
    msibi.utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |citation| image:: https://img.shields.io/badge/DOI-10.1063%2F1.4880555-blue.svg
    :target: http://dx.doi.org/10.1063/1.4880555
