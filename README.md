<img src="/docs/images/msibi.png" height="300">

# MultiState Iterative Boltzmann Inversion (MS-IBI)
----------------------------------------
[![pytest](https://github.com/cmelab/msibi/actions/workflows/pytest.yml/badge.svg)](https://github.com/cmelab/msibi/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/cmelab/msibi/branch/main/graph/badge.svg?token=7NFPBMBN0I)](https://codecov.io/gh/cmelab/msibi)

A package to help you manage and run pair potential optimizations using multistate iterative Boltzmann inversion.


### Install from source:
```bash
git clone https://github.com/cmelab/msibi.git
cd msibi
conda env create -f environment.yml
conda activate msibi
pip install .
```

### Using MSIBI
The MSIBI package is designed to be very object oriented. Any force optimization runs requires at least one `msibi.state.State` instance, `msibi.force.Force` instance and `msibi.optimize.MSIBI` instance. More state and forces can be added as needed.

MSIBI uses [Hoomd-Blue](https://hoomd-blue.readthedocs.io/en/latest/) to run optimization simulations. It is not required that you be familiar with Hoomd to use MSIBI as the simulation script for Hoomd is automatically generated. However, it is required that you pass in the choice of [Hoomd method](https://hoomd-blue.readthedocs.io/en/latest/module-md-methods.html), [Hoomd neighbor list](https://hoomd-blue.readthedocs.io/en/latest/module-md-nlist.html), and [Hoomd thermostat](https://hoomd-blue.readthedocs.io/en/latest/module-md-methods-thermostats.html) 

#### Example: Single state, single force
- Here is an example of learning a pair potential using a single state point with only one bead type.

```python
import hoomd
from msibi import MSIBI, State, Pair 

optimizer = MSIBI(
	nlist=hoomd.md.nlist.Cell,
	integrator=hoomd.md.methods.ConstantVolume,
	thermostat=hoomd.md.methods.thermostats.MTTK,
	dt=0.0001,
	gsd_period=int(1e4)
)

# Create a State instance, pass in a path to the target trajectory
stateA = State(name="A", kT=2.0, traj_file="stateA.gsd", alpha=1.0, n_frames=50)
optimizer.add_state(stateA)

# Create a Pair instance to be optimized
pairAA = Pair(type1="A", type2="A", optimize=True, r_cut=3.0, nbins=100) 
# Call the set_lj() method to set an initial guess potential
pairAA.set_lj(r_min=0.001, r_cut=3.0, epsilon=1.0, sigma=1.0)
optimizer.add_force(pairAA)

# Run 20 MSIBI iterations
optimizer.run_optimization(n_steps=2e6, n_iterations=20)
pairAA.save_potential("AA_final.csv")
```

#### Example: Multiple states, multiple forces
- Here is an example of learning a pair potential using multiple state points.
- In this example, we set fixed bond and angle potentials which are assumed to have been learned from previous MSIBI runs.

```python
import hoomd
from msibi import MSIBI, State, Pair, Bond, Angle 

optimizer = MSIBI(
	nlist=hoomd.md.nlist.Cell,
	integrator=hoomd.md.methods.ConstantVolume,
	thermostat=hoomd.md.methods.thermostats.MTTK,
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

# Add bond and angle forces learned from previous MSIBI runs
bondAA = Bond(type1="A", type2="A", optimize=False)
bondAA.set_from_file("bondAA.csv")
optimize.add_force(bondAA)

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
pairAA.save_potential("AA_final.csv")
```


### Citation [![Citing MSIBI](https://img.shields.io/badge/DOI-10.1063%2F1.4880555-blue.svg)](http://dx.doi.org/10.1063/1.4880555)
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
