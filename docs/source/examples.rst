.. _examples:

============
Examples
============

Below are some examples of how to use ``msibi``.
These are examples of the API only, and all files shown are place holders. See the :doc:`tutorials` for instructions on accessing a minimal provided dataset needed to work through an ``msibi`` tutorial.

Quick Example: Optimizing bond-stretching
-----------------------------------------
Here is a simple example using MSIBI to learn a bond-stretching force from a single state point:

.. code-block:: python

  import hoomd
  from msibi import MSIBI, State, Bond

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

Example 2: Multiple states, multiple forces
-------------------------------------------
- In this example, we set fixed bond and angle potentials that are included during iteration simulations.
- The bond potential will set a fixed harmonic force, while the angle potential will be set from a table potential file.
- This illustrates a use case of stringing together multiple MSIBI optimizations.
- For example, one MSIBI optimization can be used to learn and obtain a coarse-grained angle potential table file which can then be set and held fixed while learning pair potentials in a subsequent MSIBI optimization.

.. code-block:: python

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
