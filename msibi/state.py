import os
import shutil
from typing import Union
import warnings

import gsd.hoomd
import hoomd


class State(object):
    """
    A single state used as part of a multistate optimization.

    Parameters
    ----------
    name : str
        State name used in creating state directory space and output files.
    kT : (Union[float, int])
        Unitless heat energy (product of Boltzmann's constant and temperature).
    traj_file : path to a gsd.hoomd file
        The gsd trajectory associated with this state.
        This trajectory calcualtes the target distributions used
        during optimization.
    alpha : (Union[float, int]), default 1.0
        The alpha value used to scale the weight of this state.

    Attributes
    ----------
    name : str
        State name
    kT : float
        Unitless heat energy (product of Boltzmann's constant and temperature).
    traj_file : path
        Path to the gsd trajectory associated with this state.
    alpha : float
        The alpha value used to scaale the weight of this state.
    dir : str
        Path to where the State info with be saved.
    query_traj : str
        Path to the query trajectory that is created during each iteration.

    """

    def __init__(
        self,
        name: str,
        kT: float,
        traj_file: str,
        n_frames: int,
        alpha: float=1.0,
        exclude_bonded: bool=True, #TODO: Do we use this here or in Force?
        _dir=None
    ):
        self.name = name
        self.kT = kT
        self.traj_file = os.path.abspath(traj_file)
        self._n_frames = n_frames
        self._opt = None
        self._alpha = float(alpha)
        self.dir = self._setup_dir(name, kT, dir_name=_dir)
        self.query_traj = os.path.join(self.dir, "query.gsd")
        self.exclude_bonded = exclude_bonded

    def __repr__(self):
        return (
                f"{self.__class__}; "
                + f"Name: {self.name}; "
                + f"kT: {self.kT}; "
                + f"Weight: {self.alpha}"
        )

    @property
    def n_frames(self) -> int:
        """The number of frames used in calculating distributions."""
        return self._n_frames

    @n_frames.setter
    def n_frames(self, value: int):
        self._n_frames = value

    @property
    def alpha(self) -> Union[int, float]:
        """State point weighting value."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value

    def _run_simulation(
            self,
            n_steps: int,
            forces: list,
            integrator_method: str,
            method_kwargs: dict,
            thermostat: str,
            thermostat_kwargs: dict,
            dt: float,
            seed: int,
            iteration: int,
            gsd_period: int,
            backup_trajectories: bool=False
    ) -> None:
        """Run the hoomd 4 script used to run each query simulation.
        This method is called in msibi.optimize.

        """
        device = hoomd.device.auto_select()
        sim = hoomd.simulation.Simulation(device=device)
        print(f"Starting simulation {iteration} for state {self}")
        print(f"Running on device {device}")

        with gsd.hoomd.open(self.traj_file, "r") as traj:
            last_snap = traj[-1]
        sim.create_state_from_snapshot(last_snap)
        integrator = hoomd.md.Integrator(dt=dt)
        integrator.forces = forces
        thermostat = thermostat(kT=self.kT, **thermostat_kwargs)
        integrator.methods.append(
                integrator_method(
                    filter=hoomd.filter.All(),
                    thermostat=thermostat,
                    **method_kwargs
                )
        )
        sim.operations.add(integrator)
        #Create GSD writer
        gsd_writer = hoomd.write.GSD(
                filename=self.query_traj,
                trigger=hoomd.trigger.Periodic(int(gsd_period)),
                mode="wb",
        )
        sim.operations.writers.append(gsd_writer)
        # Run simulation
        sim.run(n_steps)
        gsd_writer.flush()
        if backup_trajectories:
            shutil.copy(
                    self.query_traj,
                    os.path.join(self.dir, f"query{iteration}.gsd")
            )
        print(f"Finished simulation {iteration} for state {self}")
        print()

    def _setup_dir(self, name, kT, dir_name=None) -> str:
        """Create a state directory each time a new State is created."""
        if dir_name is None:
            if not os.path.isdir("states"):
                os.mkdir("states")
            dir_name = os.path.join("states", f"{name}_{kT}")
        else:
            if not os.path.isdir(
                    os.path.join(dir_name, "states")
                    ):
                os.mkdir(os.path.join(dir_name, "states"))
            dir_name = os.path.join(dir_name, "states", f"{name}_{kT}")
        try:
            assert not os.path.isdir(dir_name)
            os.mkdir(dir_name)
        except AssertionError:
            print(f"{dir_name} already exists")
            raise
        return os.path.abspath(dir_name)
