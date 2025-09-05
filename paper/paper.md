---
title: 'MSIBI: Multistate Iterative Boltzmann Inversion'
tags:
  - molecular simulation
  - materials science
  - molecular dynamics
  - coarse-graining
authors:
  - name: Chris Jones
    orcid: 0000-0002-6196-5274
    equal-contrib: true
    affiliation: 1
  - name: Mazin Almarashi
    orcid: 0009-0008-1476-1237
    equal-contrib: false
    affiliation: 1
  - name: Clare M$^c$Cabe
    orcid: 0000-0002-3267-1410
    corresponding: true
    affiliation: 1
affiliations:
 - name: School of Engineering and Physical Sciences, Heriot-Watt University, Scotland, United Kingdom
   index: 1
date: 10 September 2025
bibliography: paper.bib
---

# Summary

Iterative Boltzmann inversion (IBI) is a well-established, and widely used, method for deriving coarse-grained force fields
that recreate the structrual distributions of an underlying atomistic model.
Multiple state IBI (MSIBI) as introduced by Moore et al. [@moore2014], addressse state-point transerability limitations of IBI by including distributions from multiple
state points to inform the derived coarse-grained force field.
`msibi` is a pure python package that implements the multi-state iterative Boltzmann inversion (MSIBI) method for deriving coarse-grain molecular dynamics
potentials for both intramolecular and intermolecular interactions.
The package offers a user-friendly, Python-native API, eliminating the need for bash scripting and manual editing of input files.
The intuitive API enables streamlined workflows for creating a set of potentials--such as bond stretching, bending, torsions, and non-bonded pairs--
that together make a complete coarse-grained forcefield.
`msibi` is ultimately simulation engine agnostic, but uses the HOOMD-Blue simulation engine[@anderson2020hoomd] under-the-hood to perform
iterative coarse-grained simulations. This means that `msibi` can utilize graphical processing units (GPUs) acceleration without
requiring users to manually compile GPU compatible code.

# Statement of need

Molecular dynamics (MD) simulations are computationally expensive and scale with the number of particles $(N)$ at best $O(N log N)$
which limits accessible time and length scales.
As a result, atomistic MD simulations of complex systems such as polymers and biomolecules become prohibitively expensive,
especially as their relevant length and time scales often surpass micrometers and microseconds, respectively.
Coarse-graining (CG) is a commonly adopted solution to this challenge as it reduces computational cost by grouping--or mapping--atoms into
single, larger beads.
However, this approach introduces two challenges: First, the potential energy surface for a given CG mapping is not known a priori, and
second, as the mapping used is arbitrary, with multiple valid options, development of a single CG forcefield that is transferable across mapping choices is not possible.
Therefore, developing a CG force field is required each time a new under-lying chemistry and mapping is used.
CG force fields can be derived through top-down methods, where parameters are tuned to match material properties, or bottom-up methods, where parameters
are derived to reproduce properties of a fine-grained (i.e., target) simulation.
Iterative Boltzmann inversion, and therefore MSIBI, are bottom-up methods that reproduce structural distributions of a target simulation.



Despite the wide-spread use of coarse-grianing in molecular modeling, and development of multiple methods to derive CG force fields, open-source software tools
that make this task approachable and efficient are not widely available.

Talk about Martini. Successful model, but limits mapping choices and underlying chemistry. Great tool for the targeted applicaiton, but limits on transerability.

Talk about VOTCA (?) IBI tool. Doesn't hangle multiple states, and requires bash scripting and maybe some input files?



# Using MSIBI

Basic data classes

Forces can be set and held fixed, or set and updated. This is key for building up towards a complete CG forcefeidl (intermolecular and intramolecular)
but also makes it possible to mix CG models. For example, learning the CG forces for a solvent molecule once, then using that to derive solvent-solute
non-bonded pair potentials for any number of solutes.

# Availability
`msibi` is open-source and freely available under the MIT License
on [GitHub](https://github.com/mosdef-hub/msibi). For installation instructions, and Python API documentation
please visit the [documentation](https://msibi.readthedocs.io/en/latest/).
For examples of how to use `msibi`,
please visit the [tutorials](https://msibi.readthedocs.io/en/latest/tutorials.html).

# Acknowledgements


# Conflict of Interest Statement
The authors declare the absence of any conflicts of interest: No author has any financial,
personal, professional, or other relationship that affect our objectivity toward this work.

# References
