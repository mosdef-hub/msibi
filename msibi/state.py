import os
import shutil
from typing import Union

import gsd.hoomd
import hoomd
import numpy as np

from msibi.potentials import alpha_array


class State(object):
    """A single state used as part of a multistate optimization.

    Parameters
    ----------
    name : str
        State name used in creating state directory space and output files.
    kT : (Union[float, int])
        Unitless heat energy (product of Boltzmann's constant and temperature).
    traj_file : path to a gsd.hoomd file
        The target gsd trajectory associated with this state.
        This trajectory calcualtes the target distributions used
        during optimization.
    n_frames : int, required
        The number of frames to use when calculating distributions.
        When calculating distributions, the last `n_frames` of the
        trajectory will be used.
    alpha0 : (Union[float, int]), default 1.0
        The base alpha value used to scale the weight of this state.
    alpha_form: str, optional, default 'constant'
        Alpha can be a constant number that is applied to the potential at all
        independent values (x), or it can be a linear function that approaches
        zero as x approaches x_cut.
    exclude_bonded: bool, optional, default `False`
        If `True` then any beads that belong to the same molecle
        are not included in radial distribution funciton calculations.
    """

    def __init__(
        self,
        name: str,
        kT: float,
        traj_file: str,
        n_frames: int,
        alpha0: float = 1.0,
        alpha_form: str = "constant",
        exclude_bonded: bool = False,  # TODO: Do we use this here or in Force?
        _dir=None,
    ):
        if alpha_form.lower() not in ["constant", "linear"]:
            raise ValueError(
                "The only supported alpha forms are `constant` and `linear`"
            )
        self.name = name
        self.kT = kT
        self.traj_file = os.path.abspath(traj_file)
        self._n_frames = n_frames
        self._opt = None
        self._alpha0 = float(alpha0)
        self.alpha_form = alpha_form
        self.dir = self._setup_dir(name, kT, dir_name=_dir)
        self.query_traj = os.path.join(self.dir, "query.gsd")
        self.exclude_bonded = exclude_bonded

    def __repr__(self):
        return (
            f"{self.__class__}; "
            + f"Name: {self.name}; "
            + f"kT: {self.kT}; "
            + f"Alpha0: {self.alpha0}"
        )

    @property
    def n_frames(self) -> int:
        """The number of frames used in calculating distributions."""
        return self._n_frames

    @n_frames.setter
    def n_frames(self, value: int):
        """Set the number of frames to use for calculating distributions."""
        self._n_frames = value

    @property
    def alpha0(self) -> Union[int, float]:
        """State point base weighting value."""
        return self._alpha0

    @alpha0.setter
    def alpha0(self, value: float):
        """Set the value of alpha0 for this state."""
        if value < 0:
            raise ValueError("alpha0 must be equal to or larger than zero.")
        self._alpha0 = value

    def alpha(
        self, pot_x_range: np.ndarray = None, dx: float = None
    ) -> Union[float, np.ndarray]:
        """State point weighting value.

        Parameters
        ----------
        pot_x_range : np.ndarray, optional, default = None
            The x value range for the potential being optimized.
            This is used to generate an array of alpha values, so
            must be defined when msibi.State.alpha_form is "linear".
        """
        if self.alpha_form == "constant":
            return self.alpha0
        else:
            if pot_x_range is None or dx is None:
                raise ValueError(
                    "A potential's x value range must be "
                    "given when an msibi.State.state is using "
                    "an alpha form that is not `constant`."
                )
            return alpha_array(
                alpha0=self.alpha0,
                pot_r=pot_x_range,
                dr=dx,
                form=self.alpha_form,
            )

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
        backup_trajectories: bool = False,
    ) -> None:
        """The Hoomd 4 script used to run each query simulation.
        This method is called in msibi.optimize().
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
                **method_kwargs,
            )
        )
        sim.operations.add(integrator)
        # Create GSD writer
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
                self.query_traj, os.path.join(self.dir, f"query{iteration}.gsd")
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
            if not os.path.isdir(os.path.join(dir_name, "states")):
                os.mkdir(os.path.join(dir_name, "states"))
            dir_name = os.path.join(dir_name, "states", f"{name}_{kT}")
        try:
            assert not os.path.isdir(dir_name)
            os.mkdir(dir_name)
        except AssertionError:
            print(f"{dir_name} already exists")
            raise
        return os.path.abspath(dir_name)
