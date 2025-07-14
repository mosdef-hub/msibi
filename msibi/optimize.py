import pickle

import hoomd

import msibi


class MSIBI(object):
    """Management class for orchestrating an MSIBI optimization.

    .. note::

        This package is very object oriented. This class should be
        created first, followed by at least one :class:`msibi.state.State`
        instance and at least one :class:`msibi.forces.Force` instance.

        Then, use the methods :meth:`MSIBI.add_state` and :meth:`MSIBI.add_force`
        before beginning the optimization runs using :meth:`MSIBI.run_optimization`.

        For detailed examples, see the :ref:`examples` section in the documentation.

    .. note::

        While using MSIBI does not require thorough knowledge of HOOMD-blue's API,
        some familiarity is needed to set simulation parameters such as the
        neighbor list, integrator method, thermostat, etc.

        See the HOOMD-blue documentation:
        https://hoomd-blue.readthedocs.io/en/latest/


    Parameters
    ----------
    nlist : hoomd.md.nlist.NeighborList
        The type of Hoomd neighbor list to use.
    integrator_method : hoomd.md.methods.Method
        The integrator method to use in the query simulation.
        The only supported options are ConstantVolume or ConstantPressure.
    integrator_kwargs : dict
        The arguments and their values required by the integrator chosen
    thermostat : hoomd.md.methods.thermostat.Thermostat
        The thermostat to be paired with the integrator method.
    thermostat_kwargs : dict
        The arguments and their values required by the thermostat chosen.
    dt : float, required
        The time step delta
    gsd_period : int
        The number of frames between snapshots written to query.gsd
    n_steps : int
        How many steps to run the query simulations
    nlist_exclusions : list of str, default = ["1-2", "1-3"]
        Sets the pair exclusions used during the optimization simulations
    seed : int, default=42
        Random seed to use during the simulation
    """

    def __init__(
        self,
        nlist: hoomd.md.nlist.NeighborList,
        integrator_method: hoomd.md.methods.Method,
        thermostat: hoomd.md.methods.thermostats.Thermostat,
        method_kwargs: dict,
        thermostat_kwargs: dict,
        dt: float,
        gsd_period: int,
        nlist_exclusions: list[str] = ["bond", "angle"],
        seed: int = 42,
    ):
        if integrator_method not in [
            hoomd.md.methods.ConstantVolume,
            hoomd.md.methods.ConstantPressure,
        ]:
            raise ValueError(
                "MSIBI is only compatible with NVT "
                "(hoomd.md.methods.ConstantVolume), or NPT "
                "(hoomd.md.methods.ConstantPressure)"
            )
        self.nlist = nlist
        self.integrator_method = integrator_method
        self.thermostat = thermostat
        self.method_kwargs = method_kwargs
        self.thermostat_kwargs = thermostat_kwargs
        self.dt = dt
        self.gsd_period = gsd_period
        self.seed = seed
        self.nlist_exclusions = nlist_exclusions
        self.n_iterations = 0
        self.states = []
        self.forces = []
        self._optimize_forces = []

    def add_state(self, state: msibi.state.State) -> None:
        """Add a state point to be included in query simulations.

        Parameters
        ----------
        state : msibi.state.State
            A state point to be included in query simulations

        .. note::
            At least one state point must be added before optimization can occur.
        """
        # TODO: Do we still need the ._opt attr?
        state._opt = self
        self.states.append(state)

    def add_force(self, force: msibi.forces.Force) -> None:
        """Add a force to be included in the query simulations.

            Parameters
            ----------
            force : msibi.forces.Force
                A force to be included in query simulations

        .. note::
            Only one type of force can be optimized at a time.
            Forces not set to be optimized are held fixed during query simulations.
        """
        self.forces.append(force)
        if force.optimize:
            self._add_optimize_force(force)
        for state in self.states:
            force._add_state(state)

    @property
    def bonds(self) -> list:
        """All instances of :class:`msibi.forces.Bond` included in query simulations."""
        return [f for f in self.forces if isinstance(f, msibi.forces.Bond)]

    @property
    def angles(self) -> list:
        """All instances of :class:`msibi.forces.Angle` included in query simulations."""
        return [f for f in self.forces if isinstance(f, msibi.forces.Angle)]

    @property
    def pairs(self) -> list:
        """All instances of :class:`msibi.forces.Pair` included in query simulations."""
        return [f for f in self.forces if isinstance(f, msibi.forces.Pair)]

    @property
    def dihedrals(self) -> list:
        """All instances of :class:`msibi.forces.Dihedral` included in query simulations."""
        return [f for f in self.forces if isinstance(f, msibi.forces.Dihedral)]

    def _add_optimize_force(self, force: msibi.forces.Force) -> None:
        """Check that all forces to be optimized are the same type."""
        if not all(
            [isinstance(force, f.__class__) for f in self._optimize_forces]
        ):
            raise RuntimeError(
                "Only one type of force (i.e., Bonds, Angles, Pairs, etc) "
                "can be set to optimize at a time."
            )
        self._optimize_forces.append(force)

    def run_optimization(
        self,
        n_steps: int,
        n_iterations: int,
        backup_trajectories: bool = False,
        _dir=None,
    ) -> None:
        """Runs query simulations and performs MSIBI
            on the forces set to be optimized.

            Parameters
            ----------
            n_steps : int
                Number of simulation steps during each iteration.
            n_iterations : int
                Number of MSIBI update iterations.
            backup_trajectories : bool, default=False
                If ``True``, copies of the query simulation trajectories
                are saved in their respective :class:`msibi.state.State` directory.

        .. tip::

            This method can be called multiple times, and the optimization will continue
            from the last iteration. This may be useful for inspecting the f-fit score
            before running more iterations, or smoothing the potential between iterations.
        """
        for n in range(n_iterations):
            print(f"---Optimization: {n+1} of {n_iterations}---")
            forces = self._build_force_objects()
            for state in self.states:
                state._run_simulation(
                    n_steps=n_steps,
                    forces=forces,
                    integrator_method=self.integrator_method,
                    method_kwargs=self.method_kwargs,
                    thermostat=self.thermostat,
                    thermostat_kwargs=self.thermostat_kwargs,
                    dt=self.dt,
                    seed=self.seed,
                    iteration=self.n_iterations,
                    gsd_period=self.gsd_period,
                    backup_trajectories=backup_trajectories,
                )
            self._update_potentials()
            self.n_iterations += 1

    def pickle_forces(self, file_path: str) -> None:
        """Save the HOOMD-Blue force objects for all forces to a single pickle file.

            Parameters
            ----------
            file_path : str
                The path and file name for the pickle file.

        .. tip::
            Use this method as a convienent way to save and use the final
            set of forces to be used in your own HOOMD-Blue script.
        """
        forces = self._build_force_objects()
        if len(forces) == 0:
            raise RuntimeError(
                "No forces have been created yet. See MSIBI.add_force()"
            )
        f = open(file_path, "wb")
        pickle.dump(forces, f)

    def _build_force_objects(self) -> list:
        """Creates force objects for query simulations."""
        # Create pair objects
        pair_force = None
        for pair in self.pairs:
            if not pair_force:  # Only create hoomd.md.pair obj once
                pair_force = hoomd.md.pair.Table(
                    nlist=self.nlist(
                        buffer=0.4,
                        exclusions=self.nlist_exclusions,
                        default_r_cut=0,
                    )
                )
            pair_force.params[pair._pair_name] = pair._table_entry()
            pair_force.r_cut[pair._pair_name] = pair.r_cut

        # Create bond objects
        bond_force = None
        for bond in self.bonds:
            if not bond_force:
                hoomd_bond_force = getattr(hoomd.md.bond, bond.force_init)
                if bond.force_init == "Table":
                    bond_force = hoomd_bond_force(width=bond.nbins + 1)
                else:
                    bond_force = hoomd_bond_force()
            if bond.format == "table":
                bond_force.params[bond.name] = bond._table_entry()
            else:
                bond_force.params[bond.name] = bond.force_entry

        # Create angle objects
        angle_force = None
        for angle in self.angles:
            if not angle_force:
                hoomd_angle_force = getattr(hoomd.md.angle, angle.force_init)
                if angle.force_init == "Table":
                    angle_force = hoomd_angle_force(width=angle.nbins + 1)
                else:
                    angle_force = hoomd_angle_force()
            if angle.format == "table":
                angle_force.params[angle.name] = angle._table_entry()
            else:
                angle_force.params[angle.name] = angle.force_entry
        # Create dihedral objects
        dihedral_force = None
        for dih in self.dihedrals:
            if not dihedral_force:
                hoomd_dihedral_force = getattr(
                    hoomd.md.dihedral, dih.force_init
                )
                if dih.force_init == "Table":
                    dihedral_force = hoomd_dihedral_force(width=dih.nbins + 1)
                else:
                    dihedral_force = hoomd_dihedral_force()
            if dih.format == "table":
                dihedral_force.params[dih.name] = dih._table_entry()
            else:
                dihedral_force.params[dih.name] = dih.force_entry
        forces = [pair_force, bond_force, angle_force, dihedral_force]
        return [f for f in forces if f]  # Filter out any None values

    def _update_potentials(self) -> None:
        """Update the potentials for the potentials to be optimized."""
        for force in self._optimize_forces:
            self._recompute_distribution(force)
            force._update_potential()

    def _recompute_distribution(self, force: msibi.forces.Force) -> None:
        """Recompute the current distribution."""
        for state in self.states:
            force._compute_current_distribution(state)
            force._save_current_distribution(state, iteration=self.n_iterations)
            print(
                "Force: {0}, State: {1}, Iteration: {2}: {3:f}".format(
                    force.name,
                    state.name,
                    self.n_iterations,
                    force._states[state]["f_fit"][self.n_iterations],
                )
            )
            print()
