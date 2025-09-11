<img src="/docs/images/msibi.png" height="300">

# MultiState Iterative Boltzmann Inversion (MS-IBI)
----------------------------------------
[![pytest](https://github.com/mosdef-hub/msibi/actions/workflows/CI.yaml/badge.svg)](https://github.com/mosdef-hub/msibi/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh/mosdef-hub/msibi/branch/main/graph/badge.svg?token=7NFPBMBN0I)](https://codecov.io/gh/mosdef-hub/msibi)
[![Citing MSIBI](https://img.shields.io/badge/DOI-10.1063%2F1.4880555-blue.svg)](http://dx.doi.org/10.1063/1.4880555)

A package to help you manage and run coarse-grain potential optimizations using multistate iterative Boltzmann inversion.

## Installing MSIBI

### Install from conda-forge (Coming soon):
```
mamba install -c conda-forge msibi
```

### Install from source:
```bash
git clone https://github.com/mosdef-hub/msibi.git
cd msibi
mamba env create -f environment.yml
mamba activate msibi
pip install .
```

## Using MSIBI
For a full description of the API with examples see the [documentation](https://msibi.readthedocs.io/en/latest/).

The MSIBI package is designed to be very object oriented. Any force optimization runs requires at least one `msibi.state.State` instance, `msibi.force.Force` instance and `msibi.optimize.MSIBI` instance. More state and forces can be added as needed. Multiple forces can be added with some held fixed while others are being optimized after each iteation. MSIBI is designed to allow for optimization of both intra-molecular and inter-molecular potentials.

MSIBI uses [HOOMD-Blue](https://hoomd-blue.readthedocs.io/en/latest/) to run optimization simulations. It is not required that the target (i.e., atomistic) simulations use HOOMD-Blue. Also, it is not required that you be familiar with HOOMD to use MSIBI as the simulation script is automatically generated and ran. However, it is required that you pass in the choice of [method](https://hoomd-blue.readthedocs.io/en/latest/module-md-methods.html), [neighbor list](https://hoomd-blue.readthedocs.io/en/latest/module-md-nlist.html), and [thermostat](https://hoomd-blue.readthedocs.io/en/latest/module-md-methods-thermostats.html) to the `msibi.optimize.MSIBI` class. Since MSIBI utilizes Hoomd-Blue, this means that MSIBI can run on GPUs, see [Hoomd's installation guide](https://hoomd-blue.readthedocs.io/en/latest/installation.html) for instructions on ensuring your environment includes a GPU build of hoomd. The resulting coarse-grained potentials are exported in a tabulated format compatible with other simulation engines such as LAMMPS and GROMACS.

### Quick Example: Optimizing bond-stretching
Here is a simple example using MSIBI to learn a bond-stretching force from a single state point:

```python
# This is the context/management class for MSIBI
# Set simulation parameters, call `add_state` and `add_force` methods to store other MSIBI objects.
optimizer = MSIBI(
	nlist=hoomd.md.nlist.Cell,
    integrator_method=hoomd.md.methods.ConstantVolume,
	thermostat=hoomd.md.methods.thermostats.MTTK,
    thermostat_kwargs={"tau": 0.1},
    method_kwargs={},
	dt=0.0001,
	gsd_period=int(1e4)
)

# Create a State instance, pass in a path to the target trajectory
stateA = State(name="A", kT=5.0, traj_file="cg_trajectory.gsd", alpha0=0.7, n_frames=100)

# For each force you want to optimize, create an instance, set optimize=True
AA_bond = Bond(type1="A", type2="A", optimize=True, nbins=80)
AA_bond.set_polynomial(x_min=0.0, x_max=0.5, x0=0.22, k2=5000, k3=0, k4=0)
AB_bond = Bond(type1="A", type2="B", optimize=True, nbins=80)
AB_bond.set_polynomial(x_min=0.0, x_max=0.5, x0=0.22, k2=5000, k3=0, k4=0)
# Add all states and forces to the optimization class (MSIBI)
optimizer.add_state(stateA)
optimizer.add_force(AA_bond)
optimizer.add_force(AB_bond)
optimizer.run_optimization(n_iterations=10, n_steps=2e5)

# See distribution comparison
AA_bond.plot_distribution_comparison(state=stateA)
AB_bond.plot_distribution_comparison(state=stateA)

# Save potentials
AA_bond.save_potential("AA_bond.csv")
AB_bond.save_potential("AB_bond.csv")
```

### Quick Example: Multiple states, multiple forces
- Here is an example of learning a pair potential using multiple state points and forces.
- In this example, we set fixed bond and angle potentials that are included during iteration simulations.
- The bond potential will set a fixed harmonic force, while the angle potential will be set from a table potential file.
- This illustrates a use case of stringing together multiple MSIBI optimizations.
- For example, one MSIBI optimization can be used to learn and obtain a coarse-grained angle potential table file which can then be set and held fixed while learning pair potentials in a subsequent MSIBI optimization.

```python
import hoomd
from msibi import MSIBI, State, Pair, Bond, Angle

optimizer = MSIBI(
	nlist=hoomd.md.nlist.Cell,
	integrator_method=hoomd.md.methods.ConstantVolume,
	thermostat=hoomd.md.methods.thermostats.MTTK,
	thermostat_kwargs={"tau": 0.1},
	method_kwargs={},
	dt=0.0001,
	gsd_period=int(1e4)
)

# Create 3 State instances, pass in a path to the target trajectory
stateA = State(name="A", kT=2.0, traj_file="stateA.gsd", alpha=0.2, n_frames=50)
stateB = State(name="B", kT=4.0, traj_file="stateB.gsd", alpha=0.5, n_frames=50)
stateC = State(name="C", kT=6.0, traj_file="stateC.gsd", alpha=0.3, n_frames=50)
optimizer.add_state(stateA)
optimizer.add_state(stateB)
optimizer.add_state(stateC)

# Add bond and set a harmonic force (e.g. fit to Boltzmann inverse of the distribtion)
bondAA = Bond(type1="A", type2="A", optimize=False)
bondAA.set_harmonic(r0=1.4, k=800)
optimize.add_force(bondAA)

# Add angle and load previously obtained table potential
angleAA = Angle(type1="A", type2="A", type3="A", optimize=False)
angleAA.set_from_file("angleAA.csv")
optimize.add_force(angleAA)

# Create a Pair instance to be optimized.
pairAA = Pair(type1="A", type2="A", optimize=True, r_cut=3.0, nbins=100)
# Call the set_lj() method to set an initial guess potential
pairAA.set_lj(r_min=0.001, r_cut=3.0, epsilon=1.0, sigma=1.0)
optimizer.add_force(pairAA)

# Run 20 MSIBI iterations
optimizer.run_optimization(n_steps=2e6, n_iterations=20)
pairAA.save_potential("pairAA.csv")
```


## Citing MSIBI
Details of the underlying method and its validation can be found [here](http://dx.doi.org/10.1063/1.4880555).

If you use this package, please cite the above paper. The BibTeX reference is
```
@article{Moore2014,
      author = "Moore, Timothy C. and Iacovella, Christopher R. and McCabe, Clare",
      title = "Derivation of coarse-grained potentials via multistate iterative Boltzmann inversion",
      journal = "The Journal of Chemical Physics",
      year = "2014",
      volume = "140",
      number = "22",
      doi = "http://dx.doi.org/10.1063/1.4880555"
}
```
