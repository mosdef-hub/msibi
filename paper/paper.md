---
title: 'MSIBI: Multistate Iterative Boltzmann Inversion'
tags:
  - molecular simulation
  - materials science
  - molecular dynamics
  - coarse-graining
authors:
  - name: Chris D. Jones
    orcid: 0000-0002-6196-5274
    affiliation: 1
  - name: Mazin Almarashi
    orcid: 0009-0008-1476-1237
    affiliation: 1
  - name: Marjan Albooyeh
    orcid: 0009-0001-9565-3076
    affiliation: 2
  - name: Clare McCabe
    orcid: 0000-0002-3267-1410
    corresponding: true
    affiliation: 1
affiliations:
 - name: School of Engineering and Physical Sciences, Heriot-Watt University, Edinburgh, Scotland, United Kingdom
   index: 1
 - name: Micron School of Material Science and Engineering, Boise State University, Boise, Idaho, United States
   index: 2
date: 12 September 2025
bibliography: paper.bib
---

# Summary

Iterative Boltzmann inversion (IBI) is a well-established, and widely used, method for deriving coarse-grained force fields that recreate the structural distributions of an underlying atomistic model.
Multiple state IBI (MSIBI) as introduced by Moore et al. [@Moore2014], addresses state-point transferability limitations of IBI by including distributions from multiple state points to inform the derived coarse-grained force field.
Here, we introduce `msibi`, a pure python package that implements the MSIBI method for deriving coarse-grain molecular dynamics force fields for both intramolecular and intermolecular interactions.
The package offers a user-friendly, Python-native API, eliminating the need for bash scripting and manual editing of multiple input files.
`msibi` is ultimately simulation engine agnostic, but uses the HOOMD-Blue simulation engine [@Anderson2020hoomd] under-the-hood to perform iterative coarse-grained simulations.
This means that `msibi` can utilize graphical processing unit (GPU) acceleration without requiring users to manually compile GPU compatible code.

# Statement of need

Molecular dynamics (MD) simulations are computationally expensive and scale poorly with the number of particles simulated, which limits accessible time and length scales.
As a result, atomistic MD simulations of complex systems such as polymers and biomolecules become prohibitively expensive, especially as their relevant length and time scales often surpass micrometers and microseconds.
Coarse-graining (CG) is a commonly adopted solution to this challenge, as it reduces computational cost by grouping—or mapping—atoms into a single, larger bead [@Joshi2021].
However, this approach introduces two challenges: first, the potential energy surface for a given chemistry and CG mapping is not known a priori, and
second, as the mapping used is arbitrary, with multiple valid options, developing a single CG force field that is transferable across various mapping choices is not possible.
Consequently, developing a CG force field is required each time a new under-lying chemistry or mapping is chosen.
IBI and MSIBI are popular choices for deriving CG forces for polymers and biomolecules [@Carbone2008; @Moore2016; @Jones2025; @Tang2023; @Fritz2009].
While these methods are widely used, open-source software tools that provide an accessible and reproducible, end-to-end workflow for IBI and MSIBI remain limited, especially for arbitrary mappings and multi-state systems.

The MARTINI force field is a widely adopted CG model focusing on biomolecular and soft matter systems [@Martini2007].
However, it utilizes standardized mapping and bead definitions, which ensure transferability but also constrain users to predefined choices of chemistry and resolution.
Similarly, VOTCA offers a robust implementation of IBI—among several other features—and is widely used in the community [@Baumeier2024].
However, its workflow relies on manual management of multiple input files and bash operations, which can introduce operational complexity that reduces reproducibility and usability [@Cummings_2020; @Jankowski2019].
Additionally, VOTCA's implementation of IBI does not natively support inclusion and weighting of multiple state points.

Here, `msibi` is designed to execute successive CG force optimizations in series, where the learned force from the previous optimization is included and held fixed while the next force is optimized.
This follows best practices for deriving CG force fields via IBI and MSIBI [@Reith2003].
Finally, we emphasize that `msibi` is ultimately engine agnostic: any simulation engine can generate the fine-grained target structural distributions, and the CG force field produced by `msibi` is compatible with any simulation engine that supports tablulated potentials.
This includes LAMMPS [@Thompson2022], Gromacs [@Van2005], and HOOMD-Blue [@Anderson2020hoomd], among others.
It is required that the target trajectories are converted to the [gsd](https://gsd.readthedocs.io/en/v4.0.0/) file format, which is the native file format for HOOMD-Blue.
`msibi` includes a utility function that converts trajectory files from LAMMPS, Gromacs, and CHARMM into the gsd file format.
This converter utility relies on the MDAnalysis package [@Naughton2022] as a back-end to help handle file conversions.


# Using MSIBI

`msibi` contains three primary classes:

## 1. **msibi.state.State:**
This class encapsulates state-point information such as target trajectories, temperature, weighting factor and sampling parameters.
Multiple instances of this class can be created, and each is used in deriving the final CG force field.

## 2. **msibi.force.Force:**
The base class from which all force types in `msibi` inherit from:

- **msibi.force.Bond:** Optimizes bond-stretching forces.
- **msibi.force.Angle:** Optimizes bond-bending forces.
- **msibi.force.Pair:** Optimizes non-bonded pair forces.
- **msibi.force.Dihedral:** Optimizes bond-torsion forces.

Users can include any number and combination of forces for MSIBI simulations, though only one *type* of force can be optimized at a time.

### Setting Force parameters
There are multiple methods for defining the parameters of a `msibi.force.Force` instance:

- **Force.set_from_file:** Creates a tabulated force from a `.csv` file. This is useful for setting a previously optimized CG force while learning another.
- **Force.set_polynomial:** Creates a tabulated force from a polynomial function. This is helpful to setting initial guess forces, especially for distributions with multiple peaks.
- **Force.set_harmonic:** Creates a static, immutable harmonic force (not tabulated). This is useful for setting force parameters for distributions that are easily described by a harmonic function.

### Example workflow
These methods enable users to combine learned and static forces and include them in series, resulting in a single CG force field. For example:

1. Fit a bond-stretching force to a simple distribution and set the force using `Bond.set_harmonic()`.
2. With the bond-stretching force included and held static (step 1), run `msibi` to learn bond-angle forces, resulting in a tabulated force stored in a `.csv` file.
3. Set up and run a new instance where `Bond.set_harmonic()` and `Angle.set_from_file()` create static intra-molecular forces and learn a non-bonded force.

## 3. **msibi.optimize.MSIBI:**
This class serves as the context manager for orchestrating optimization iterations and ensures the correct interactions are updated.
A single instance of this class is needed, and all instances of **msibi.state.State** and **msibi.force.Force** are attached to it before optimizaitons begin.
This class also stores global simulation parameters such as timestep, neighbor list, exclusions, thermostat and trajectory write-out frequency.

### Primary methods
- `MSIBI.add_state` and `MSIBI.add_force:` Handle data management between states and forces.
- `MSIBI.run_optimization`: Runs iterative simulations and updates for all instances of `msibi.force.Force` being optimized.

The `run_optimization` method is designed for flexibility.
It can be called multiple times, resuming from the last iteration.
This enables use in:

- While loops: Run single iterations until a convergence criterion is met.

- For loops: Perform operations between batches of iterations. For example:
    - Smoothing the force
    - Adjusting state-point weighting
    - Modifying simulation criteria (e.g., extending optimization simulations as the force stabilizes).

The [repository](https://github.com/mosdef-hub/msibi) and [documentation](https://msibi.readthedocs.io/en/latest/) contain more detailed examples.

# Availability

`msibi` is open-source and freely available under the MIT License on [GitHub](https://github.com/mosdef-hub/msibi).
We encourage users to file issues and make contributions as applicable on the repository.
`msibi` is available on the conda-forge ecosystem.
For installation instructions and Python API documentation, visit the [documentation](https://msibi.readthedocs.io/en/latest/).

# Acknowledgements


# Conflict of Interest Statement
The authors declare the absence of any conflicts of interest: No author has any financial,
personal, professional, or other relationship that affect our objectivity toward this work.

# References
