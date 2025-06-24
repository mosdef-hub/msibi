<img src="/docs/images/msibi.png" height="300">

# MultiState Iterative Boltzmann Inversion (MS-IBI)
----------------------------------------
[![pytest](https://github.com/cmelab/msibi/actions/workflows/pytest.yml/badge.svg)](https://github.com/cmelab/msibi/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/cmelab/msibi/branch/main/graph/badge.svg?token=7NFPBMBN0I)](https://codecov.io/gh/cmelab/msibi)
[![Citing MSIBI](https://img.shields.io/badge/DOI-10.1063%2F1.4880555-blue.svg)](http://dx.doi.org/10.1063/1.4880555)

A package to help you manage and run coarse-grain potential optimizations using multistate iterative Boltzmann inversion.

## Installing MSIBI

### Install from conda-forge:
```
conda install -c conda-forge msibi
```

### Install from source:
```bash
git clone https://github.com/cmelab/msibi.git
cd msibi
conda env create -f environment.yml
conda activate msibi
pip install .
```

## Using MSIBI
The MSIBI package is designed to be very object oriented. Any force optimization runs requires at least one `msibi.state.State` instance, `msibi.force.Force` instance and `msibi.optimize.MSIBI` instance. More state and forces can be added as needed.

MSIBI uses [Hoomd-Blue](https://hoomd-blue.readthedocs.io/en/latest/) to run optimization simulations. It is not required that you be familiar with Hoomd to use MSIBI as the simulation script is automatically generated and ran. However, it is required that you pass in the choice of [Hoomd method](https://hoomd-blue.readthedocs.io/en/latest/module-md-methods.html), [Hoomd neighbor list](https://hoomd-blue.readthedocs.io/en/latest/module-md-nlist.html), and [Hoomd thermostat](https://hoomd-blue.readthedocs.io/en/latest/module-md-methods-thermostats.html) to the `msibi.optimize.MSIBI` instance. 

### Quick Example:
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
	gsd_period=int(1e3)
)

# Create a State instance, pass in a path to the target trajectory
stateA = State(name="A", kT=5.0, traj_file="cg_trajectory.gsd", alpha0=0.7, n_frames=100)

# For each force you want to optimize, create an instance, set optimize=True
AA_bond = Bond(type1="A", type2="A", optimize=True, nbins=80)
AA_bond.set_polynomial(x_min=0.0, x_max=0.5, x0=0.22, k2=100000, k3=0, k4=0)
AA_bond.smoothing_window = 5
AB_bond = Bond(type1="A", type2="B", optimize=True, nbins=80)
AB_bond.set_polynomial(x_min=0.0, x_max=0.5, x0=0.22, k2=100000, k3=0, k4=0)
AB_bond.smoothing_window = 5
# Add all states and forces to the optimization class (MSIBI)
optimizer.add_state(stateA)
optimizer.add_force(AA_bond)
optimizer.add_force(AB_bond)
optimizer.run_optimization(n_iterations=10, n_steps=2e5)

# See distribution comparison
AA_bond.plot_distribution_comparison(state=stateA)
AB_bond.plot_distribution_comparison(state=stateA)

AA_bond.save_potential("AA_bond.csv")
AB_bond.save_potential("AB_bond.csv")
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
