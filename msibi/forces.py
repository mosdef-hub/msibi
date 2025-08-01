import os
import warnings
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmeutils.structure import (
    angle_distribution,
    bond_distribution,
    dihedral_distribution,
    gsd_rdf,
)
from scipy.signal import savgol_filter

import msibi
from msibi.utils.corrections import (
    bonded_corrections,
    exponential,
    harmonic,
    pair_corrections,
)
from msibi.utils.exceptions import (
    PotentialImmutableError,
    PotentialNotOptimizedError,
)
from msibi.utils.general import calc_similarity, natural_sort
from msibi.utils.potentials import lennard_jones, polynomial_potential


class Force:
    """
    Base class from which all other forces inherit.
    Don't use this class directly, instead use
    :class:`Bond`, :class:`Angle`, :class:`Pair`, and :class:`Dihedral`.

    .. warning::

        This class should not be instantiated directly by users.
        It can be used for ``isinstance`` or ``issubclass``.

    .. note::

        Forces in MSIBI can either be held constant (i.e., fixed) or
        optimized (i.e., mutable).

        Only one type of force can be optimized at a time.
        For example, you can optimize multiple ``Angle`` potentials
        during one optimization run, but you cannot
        optimize a ``Pair`` and an ``Angle`` potential in the same
        optimization run.

        Several of the methods in this class are only applicable
        in the case a force is being optimized.

    Parameters
    ----------
    name : str
        The name of the type in the Force.
        Must match the names found in the State's trajectory file.
    optimize : bool
        Set to ``True`` if this force is to be mutable and optimized.
        Set to ``False`` if this force is to be held constant while
        other forces are optimized.
    nbins : int, optional
        This must be a positive integer if this force is being optimized.
        nbins is used for setting the potenials independent varible (x) range
        and step size (dx).
        It is also used in determining the bin size of the target and query
        distributions.
        If this force is not being optimied, leave this as ``None``.
    correction_fit_window: int, optional
        The window size (number of data points) to use when fitting
        the iterative potential to head and tail correction forms.
        This is only used when the Force is set to be optimized.
    correction_form: Callable, optional
        The type of correciton form to apply to the potential.
        This is only used when the Force is set to be optimized.
        Bonded forces (``Bond``, ``Angle``, ``Dihedral``) apply this correction
        to both the head and tail of the potential.
        Non-bonded forces (``Pair``) only apply this correction to the head of
        the potential.
    """

    def __init__(
        self,
        name: str,
        optimize: bool,
        nbins: Optional[int] = None,
        smoothing_window: Optional[int] = None,
        smoothing_order: Optional[int] = None,
        correction_fit_window: Optional[int] = None,
        correction_form: Optional[Callable] = None,
    ):
        if optimize and not nbins or optimize and nbins <= 0:
            raise ValueError(
                "If a force is set to be optimized, nbins must be "
                "a positive, non-zero integer."
            )
        self.name = name
        self.optimize = optimize
        self._nbins = nbins
        self.correction_fit_window = correction_fit_window if optimize else None
        self.correction_form = correction_form
        self._smoothing_window = smoothing_window if optimize else None
        self._smoothing_order = smoothing_order if optimize else None
        self.format = None
        self.xmin = None
        self.xmax = None
        self.dx = None
        self.x_range = None
        self.potential_history = []
        self._potential = None
        self._states = dict()
        self._head_correction_history = []
        self._tail_correction_history = []
        self._learned_potential_history = []

    def __repr__(self):
        return (
            f"Type: {self.__class__}; "
            + f"Name: {self.name}; "
            + f"Optimize: {self.optimize}"
        )

    @property
    def potential(self) -> np.ndarray:
        """The potential energy values V(x)."""
        if self.format != "table":
            warnings.warn(f"{self} is not using a table potential.")
            return None
        return self._potential

    @potential.setter
    def potential(self, array):
        if self.format != "table":
            raise PotentialImmutableError("set a table potential")
        self._potential = array

    @property
    def force(self) -> np.ndarray:
        """The force values F(x)."""
        if self.format != "table":
            warnings.warn(f"{self} is not using a table potential.")
            return None
        return -1.0 * np.gradient(self.potential, self.dx)

    @property
    def smoothing_window(self) -> int:
        """Window size used in smoothing the distributions and potentials."""
        return self._smoothing_window

    @smoothing_window.setter
    def smoothing_window(self, value: int):
        """Window size used in smoothing the distributions and potentials."""
        if not self.optimize:
            raise PotentialNotOptimizedError("apply a smoothing window")
        if not isinstance(value, int) or value <= 0:
            raise ValueError("The smoothing window must be an integer > 0.")
        self._smoothing_window = value
        for state in self._states:
            self._add_state(state)

    @property
    def smoothing_order(self) -> int:
        """The order used in Savitzky Golay filter."""
        return self._smoothing_order

    @smoothing_order.setter
    def smoothing_order(self, value: int):
        """The order used in Savitzky Golay filter."""
        if not self.optimize:
            raise PotentialNotOptimizedError("apply a smoothing order")
        if not isinstance(value, int) or value <= 0:
            raise ValueError("The smoothing order must be an integer.")
        self._smoothing_order = value
        for state in self._states:
            self._add_state(state)

    @property
    def nbins(self) -> int:
        """The number of bins used in distributions and x-range of the potential."""
        return self._nbins

    @nbins.setter
    def nbins(self, value: int):
        """The number of bins used in distributions and x-range of the potential."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("nbins must be an integer.")
        self._nbins = value
        for state in self._states:
            self._add_state(state)

    def smooth_potential(self) -> None:
        """Smooth and overwrite the current potential.

        .. note::

            This uses a Savitzky Golay smoothing algorithm where the
            window size and order parameters are set by
            :meth:`msibi.forces.Force.smoothing_window` and
            :meth:`msibi.forces.Force.smoothing_order`.

            Both of these can be changed using their respective setters.

        """
        if self.format != "table":
            raise PotentialNotOptimizedError("smooth the potential")
        potential = np.copy(self.potential)
        self.potential = savgol_filter(
            x=potential,
            window_length=self.smoothing_window,
            polyorder=self.smoothing_order,
        )

    def save_potential(self, file_path: str) -> None:
        """Save the x-range, potential and force to a `.csv` file.

        .. note::

            This method uses ``pandas.DataFrame.to_csv`` and saves the data
            in with column labels of "x", "potential", and "force".

            If you want to smooth the final potential, use
            :meth:`msibi.forces.Force.smooth_potential` before
            calling this method.

        Parameters
        ----------
        file_path : str
            File path and name to save table potential to.
        """
        if self.format != "table":
            raise PotentialImmutableError("save the potential to a .csv")
        df = pd.DataFrame(
            {
                "x": self.x_range,
                "potential": self.potential,
                "force": self.force,
            }
        )
        df.to_csv(file_path, index=False)

    def save_potential_history(self, file_path: str) -> None:
        """Save the potential history of the force to a `.npy` file.

        Parameters
        ----------
        file_path : str
            File path and name to save table potential history to.
        """
        if not self.optimize:
            raise PotentialNotOptimizedError("save a potential history")
        np.save(file_path, np.asarray(self.potential_history))

    def save_state_data(self, state: msibi.state.State, file_path: str) -> None:
        """Save the distribution data of a state as a a dictionary to a `.npz` file.

        .. note::

            This saves the state points target distribution, current distribution
            distribution history and f-fit scores for the corresponding potential
            to the `.npz` file.

        Parameters
        ----------
        state : msibi.state.State, required
            The state to use in finding the target distribution.
        file_path : str, required
            File path and name to save the `.npz` file.
        """
        if not self.optimize:
            raise PotentialNotOptimizedError("save a potential history")
        state_dict = self._states[state]
        state_data = {
            "target_distribution": state_dict["target_distribution"],
            "current_distribution": state_dict["current_distribution"],
            "distribution_history": np.asarray(state_dict["distribution_history"]),
            "f_fit": np.asarray(state_dict["f_fit"]),
        }
        np.savez(file_path, **state_data)

    def target_distribution(self, state: msibi.state.State) -> np.ndarray:
        """The target structural distribution corresponding to this force.

        Parameters
        ----------
        state : msibi.state.State
            The state point to use in finding the target distribution.
        """
        return self._states[state]["target_distribution"]

    def plot_target_distribution(
        self, state: msibi.state.State, file_path: str = None
    ) -> None:
        """Plot the target distribution corresponding to this force and state point.

        .. note::

            Use this to see how the shape of the target distribution is
            affected by your choices for nbins, smoothing window,
            and smoothing order.

        Parameters
        ----------
        state : msibi.state.State
            The state to use in finding the target distribution.
        file_path : str, optional
            If given, the plot will be saved to this location.
        """
        if not self.optimize:
            raise PotentialNotOptimizedError("plot a distribution history")
        target = self.target_distribution(state)
        plt.title(f"State {state.name}: {self.name} Target")
        plt.ylabel("P(x)")
        plt.xlabel("x")
        plt.plot(target[:, 0], target[:, 1], marker="^", label="Target")
        if self.smoothing_window:
            y_smoothed = savgol_filter(
                x=target[:, 1],
                window_length=self.smoothing_window,
                polyorder=self.smoothing_order,
            )
            plt.plot(target[:, 0], y_smoothed, marker="o", label="Smoothed")
            plt.legend()
        if file_path:
            plt.savefig(file_path)

    def plot_fit_scores(self, state: msibi.state.State, file_path: str = None) -> None:
        """Plot the evolution of the distribution matching fit scores.

        Parameters
        ----------
        state : msibi.state.State
            The state to use in finding the target distribution.
        file_path : str, optional
            If given, the plot will be saved to this location.
        """
        if not self.optimize:
            raise PotentialNotOptimizedError("plot fit scores")
        plt.plot(self._states[state]["f_fit"], "o-")
        plt.xlabel("Iteration")
        plt.ylabel("Fit Score")
        plt.title(f"State {state.name}: {self.name} Fit Score")
        if file_path:
            plt.savefig(file_path)

    def plot_potential(
        self,
        file_path: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
    ) -> None:
        """Plot the currently optimized potential energy.

        Parameters
        ----------
        file_path : str, optional
            If given, the plot will be saved to this location.
        xlim : tuple, optional
            If given, sets the limits for the x-range of the plot.
            If not given, uses the entire x-range used in optimization.
        ylim : tuple, optional
            If given, sets the limits for the y-range of the plot.
            If not given, uses the entire y-range of the learned potential.
        """
        if self.format != "table":
            raise PotentialImmutableError("plot the potential.")
        plt.plot(self.x_range, self.potential, "o-")
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel("x")
        plt.ylabel("V(x)")
        plt.title(f"{self.name} Potential")
        if file_path:
            plt.savefig(file_path)

    def plot_potential_history(
        self,
        file_path: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
    ) -> None:
        """Plot the history of the optimized potential energy.

        Parameters
        ----------
        file_path : str, optional
            If given, the plot will be saved to this location.
        xlim : tuple, optional
            If given, sets the limits for the x-range of the plot.
            If not given, uses the entire x-range used in optimization.
        ylim : tuple, optional
            If given, sets the limits for the y-range of the plot.
            If not given, uses the entire y-range of the learned potential.
        """
        if not self.optimize:
            raise PotentialNotOptimizedError("plot the potential history")
        for i, pot in enumerate(self.potential_history):
            plt.plot(self.x_range, pot, "o-", label=i)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.xlabel("x")
        plt.ylabel("V(x)")
        plt.title(f"{self.name} Potential History")
        if file_path:
            plt.savefig(file_path, bbox_inches="tight")

    def plot_distribution_comparison(
        self,
        state: msibi.state.State,
        file_path: Optional[str] = None,
    ) -> None:
        """Plot the target distribution and most recent query distribution.

        Parameters
        ----------
        state : msibi.state.State
            The state to use in finding the target distribution.
        file_path : str, optional
            If given, the plot will be saved to this location.
        """
        if not self.optimize:
            raise PotentialNotOptimizedError("plot the distribution comparison")
        final_dist = self.distribution_history(state=state)[-1]
        target_dist = self.target_distribution(state=state)

        plt.plot(final_dist[:, 0], final_dist[:, 1], "o-", label="MSIBI")
        plt.plot(target_dist[:, 0], target_dist[:, 1], "^-", label="Target")

        plt.legend()
        plt.xlabel("x")
        plt.ylabel("P(x)")
        if file_path:
            plt.savefig(file_path)

    def distribution_history(self, state: msibi.state.State) -> np.ndarray:
        """Get the complete query distribution history for a given state.

        Parameters
        ----------
        state : msibi.state.State
            The state point to use for calculating the distribution.

        """
        return self._states[state]["distribution_history"]

    def set_target_distribution(
        self, state: msibi.state.State, array: np.ndarray
    ) -> None:
        """Store the target distribution for a given state.

        Parameters
        ----------
        state: msibi.state.State
            The state point used in finding the distribution.
        array : np.ndarray
            The 2D array representing the target distribution for this state point.
        """
        if not self.optimize:
            raise PotentialNotOptimizedError("set a target distribution")
        self._states[state]["target_distribution"] = array

    def current_distribution(self, state: msibi.state.State) -> np.ndarray:
        """Distrubution of the most recent iteration for a given state point.

        Parameters
        ----------
        state : msibi.state.State
            The state point used for calculating the distribution.
        """
        return self._get_state_distribution(state, query=True)

    def distribution_fit(self, state: msibi.state.State) -> float:
        """Get the fit score from the most recent query simulation.

        Parameters
        ----------
        state : msibi.state.State
            The state point used for calculating the distribution.
        """
        if not self.optimize:
            raise PotentialNotOptimizedError("calculate an f-fit score")
        return self._calc_fit(state)

    def set_polynomial(
        self,
        k2: Union[float, int],
        k3: Union[float, int],
        k4: Union[float, int],
        x0: Union[float, int],
        x_min: Union[float, int],
        x_max: Union[float, int],
    ) -> None:
        """Set a potential based on the following function:

            :math:`V(x) = k4(x-x_{0})^{4} + k3(x-x_{0})^{3} + k2(x-x_{0})^{2}`

        .. note::

            Using this method will create a table potential V(x) over the range
            x_min - x_max.

            This is useful for easily setting initial guess potentials for bond, angle and dihedral
            forces to be optimized. Also see :meth:`msibi.forces.Force.set_from_file`.

            See :meth:`msibi.forces.Pair.set_lj` for setting an initial guess potential for
            a non-bonded pair force to be optimized.

        Parameters
        ----------
        x0, k2, k3, k4 : float
            The paraters used in the V(x) function described above.
        x_min : float
            The lower bound of the potential range.
        x_max : float
            The upper bound of the potential range.
        """
        self.format = "table"
        self.x_min = x_min
        self.x_max = x_max
        self.dx = x_max / self.nbins
        if isinstance(self, msibi.forces.Dihedral):
            self.dx *= 2
            self.x_range = np.arange(x_min, x_max + self.dx / 2, self.dx)
        else:
            self.x_range = np.arange(x_min, x_max + self.dx, self.dx)
        self.potential = polynomial_potential(self.x_range, x0, k4, k3, k2)
        self.force_init = "Table"
        self.force_entry = self._table_entry()

    def set_from_file(self, file_path: str) -> None:
        """Set a potential from a `.csv` file.

        .. warning::

            This uses `pandas.DataFrame.read_csv` and expects
            column names of "x", "potential", and "force".

        .. tip::

            Use this potential setter to set a potential from a previous MSIBI run.
            For example, use the final potential files from a bond-optimization IBI
            run to set a static coarse-grained bond potential while you perform
            IBI runs on angle and/or pair potentials.

            Also see: :meth:`msibi.forces.Force.save_potential`

        Parameters
        -----------
        file_path : str
            The full path to the table potential csv file.

        """
        self.format = "table"
        df = pd.read_csv(file_path)
        self.x_range = df["x"].values
        self.potential = df["potential"].values
        self.dx = np.round(self.x_range[1] - self.x_range[0], 3)
        self.x_min = self.x_range[0]
        self.x_max = self.x_range[-1] + self.dx
        self.force_init = "Table"
        self.nbins = len(self.x_range) - 1

    def _add_state(self, state: msibi.state.State) -> None:
        """Add a state to be used in optimizing this Force.

        Parameters
        ----------
        state : msibi.state.State
            Instance of a State object previously created.
        """
        if self.optimize:
            target_distribution = self._get_state_distribution(state=state, query=False)
            if self.smoothing_window and self.smoothing_order:
                target_distribution[:, 1] = savgol_filter(
                    x=target_distribution[:, 1],
                    window_length=self.smoothing_window,
                    polyorder=self.smoothing_order,
                )
        else:
            target_distribution = None
        self._states[state] = {
            "target_distribution": target_distribution,
            "current_distribution": None,
            "alpha0": state.alpha0,
            "f_fit": [],
            "distribution_history": [],
            "path": state.dir,
        }

    def _compute_current_distribution(self, state: msibi.state.State) -> None:
        """Find the current distribution of the query trajectory.

        Parameters
        ----------
        state : msibi.state.State
            Instance of a State object previously created.
        """
        distribution = self._get_state_distribution(state, query=True)
        if self.smoothing_window and self.smoothing_order:
            distribution[:, 1] = savgol_filter(
                x=distribution[:, 1],
                window_length=self.smoothing_window,
                polyorder=self.smoothing_order,
                deriv=0,
            )
            negative_idx = np.where(distribution[:, 1] < 0)[0]
            distribution[:, 1][negative_idx] = 0
        self._states[state]["current_distribution"] = distribution

        f_fit = calc_similarity(
            distribution[:, 1], self._states[state]["target_distribution"][:, 1]
        )
        self._states[state]["f_fit"].append(f_fit)

    def _get_state_distribution(
        self, state: msibi.state.State, query: bool
    ) -> np.ndarray:
        """Get the corresponding distrubiton for a given state.

        Parameters
        ----------
        state: msibi.state.State
            State used in calculating the distribution.
        query: bool
            If `True`, uses the most recent query trajectory.
            If `False`, uses the state's target trajectory.
        """
        if query:
            traj = state.query_traj
        else:
            traj = state.traj_file
        return self._get_distribution(state=state, gsd_file=traj)

    def _save_current_distribution(
        self, state: msibi.state.State, iteration: int
    ) -> None:
        """Save the corresponding distrubiton for a given `State` to a file.

        Parameters
        ----------
        state : msibi.state.State
            The state point used in finding the distribution.
        iteration : int
            Current iteration step, used in the filename.
        """
        distribution = self._states[state]["current_distribution"]
        distribution[:, 0] -= self.dx / 2
        fname = f"dist_{self.name}-state_{state.name}-step_{iteration}.txt"
        fpath = os.path.join(state.dir, fname)
        np.savetxt(fpath, distribution)

    def _update_potential(self) -> None:
        """Compare distributions and update potential via Boltzmann Inversion."""
        # TODO: TAKE THIS APPEND OUT?
        self.potential_history.append(np.copy(self.potential))
        for state in self._states:
            current_dist = self._states[state]["current_distribution"]
            target_dist = self._states[state]["target_distribution"]
            self._states[state]["distribution_history"].append(current_dist)
            N = len(self._states)
            alpha_array = state.alpha(pot_x_range=self.x_range, dx=self.dx)
            self._potential += alpha_array * (
                state.kT * np.log(current_dist[:, 1] / target_dist[:, 1]) / N
            )
        # Apply corrections to regions without distribution overlap
        if isinstance(self, msibi.forces.Pair):
            self._potential, head_cut, tail_cut, real_indices = pair_corrections(
                x=self.x_range,
                V=self.potential,
                r_switch=self.r_switch,
                smoothing_window=self.smoothing_window,
                smoothing_order=self.smoothing_order,
                fit_window_size=self.correction_fit_window,
                head_correction_func=self.correction_form,
            )
        else:  # Bonded force, use correction form on both head and tail of potential
            self._potential, head_cut, tail_cut, real_indices = bonded_corrections(
                x=self.x_range,
                V=self.potential,
                smoothing_window=self.smoothing_window,
                smoothing_order=self.smoothing_order,
                fit_window_size=self.correction_fit_window,
                head_correction_func=self.correction_form,
                tail_correction_func=self.correction_form,
            )
        # Store all versions of final potential (actually learned, and extrapolated)
        self.potential_history.append(np.copy(self.potential))
        self._head_correction_history.append(np.copy(self.potential[0:head_cut]))
        self._tail_correction_history.append(np.copy(self.potential[tail_cut:]))
        self._learned_potential_history.append(np.copy(self.potential[real_indices]))


class Bond(Force):
    """Creates a bond stretching :class:`Force` used in query simulations.

    .. note::

        The bond type is sorted so that ``type1`` and ``type2``
        are listed in alphabetical order, and must match the bond
        types found in the state's target GSD file bond types.

        For example: ``Bond(type1="B", type2="A")`` will have ``Bond.name = "A-B"``

    Parameters
    ----------
    type1 : str
        Name of the first particle type in the bond.
        This must match the types found in the state's target GSD file.
    type2 : str
        Name of the second particle type in the bond.
        This must match the types found in the state's target GSD file.
    optimize : bool
        Set to True if this force is to be mutable and optimized.
        Set to False if this force is to be held constant while
        other forces are optimized.
    nbins : int, optional
        This must be a positive integer if this force is being optimized.
        nbins is used for setting the potenials independent varible (x) range
        and step size (dx).
        It is also used in determining the bin size of the target and query
        distributions.
    correction_fit_window: int, optional
        The window size (number of data points) to use when fitting
        the iterative potential to head and tail correction forms.
        This is only used when the Force is set to be optimized.
    correction_form: Callable, default=msibi.utils.corrections.harmonic
        The type of correciton form to apply to the potential.
        This is only used when the Force is set to be optimized.
        This correction is applied to both the head and tail of the potential.
    """

    def __init__(
        self,
        type1: str,
        type2: str,
        optimize: bool,
        nbins: Optional[int] = None,
        smoothing_window: Optional[int] = None,
        smoothing_order: Optional[int] = None,
        correction_fit_window: Optional[int] = None,
        correction_form: Callable = harmonic,
    ):
        self.type1, self.type2 = sorted([type1, type2], key=natural_sort)
        name = f"{self.type1}-{self.type2}"
        super(Bond, self).__init__(
            name=name,
            optimize=optimize,
            nbins=nbins,
            smoothing_window=smoothing_window,
            smoothing_order=smoothing_order,
            correction_fit_window=correction_fit_window,
            correction_form=correction_form,
        )
        if self.optimize:
            if self.smoothing_window is None:
                self.smoothing_window = 15
            if self.smoothing_order is None:
                self.smoothing_order = 2
            if self.correction_fit_window is None:
                self.correction_fit_window = 10

    def set_harmonic(self, r0: Union[float, int], k: Union[float, int]) -> None:
        """Set a fixed harmonic bond potential.

        .. warning::

            Using this method is not compatible with :class:`Force`
            objects that are set to be optimized during MSIBI.

        .. note::

            For more information on harmonic bond potentials, refer to the
            `HOOMD-blue harmonic bond documentation <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/bond/harmonic.html>`_.

        Parameters
        ----------
        r0 : (Union[float, int])
            Equilibrium bond length [length]
        k : (Union[float, int])
            Spring constant [energy]

        """
        if self.optimize:
            raise RuntimeError(
                f"Force {self} is set to be optimized during MSIBI."
                "This potential setter cannot be used "
                "for a force designated for optimization. "
                "Instead, use `set_from_file` or `set_polynomial`."
            )
        self.format = "static"
        self.force_init = "Harmonic"
        self.force_entry = dict(r0=r0, k=k)

    def _table_entry(self) -> dict:
        """Set the correct entry to use in ``hoomd.md.bond.Table``"""
        table_entry = {
            "r_min": self.x_min,
            "r_max": self.x_max,
            "U": self.potential,
            "F": self.force,
        }
        return table_entry

    def _get_distribution(self, state: msibi.state.State, gsd_file: str) -> np.ndarray:
        """Calculate a bond length distribution.

        Parameters
        ----------
        state: msibi.state.State
            State used in calculating the distribution.
        gsd_file: str, required
            Path to the GSD file used.
        """
        return bond_distribution(
            gsd_file=gsd_file,
            A_name=self.type1,
            B_name=self.type2,
            start=-state.n_frames,
            stop=-1,
            stride=state.sampling_stride,
            histogram=True,
            normalize=True,
            as_probability=True,
            l_min=self.x_min,
            l_max=self.x_max,
            bins=self.nbins + 1,
        )


class Angle(Force):
    """Creates a bond angle :class:`Force` used in query simulations.

    .. note ::

        The angle type is formed in the order of ``type1-type2-type3``
        and must match the same order in the target GSD file angle types.

    Parameters
    ----------
    type1 : str
        Name of the first particle type in the angle.
        This must match the types found in the state's target GSD file.
    type2 : str
        Name of the second particle type in the angle.
        This must match the types found in the state's target GSD file.
    type3 : str
        Name of the third particle type in the angle.
        This must match the types found in the state's target GSD file.
    optimize : bool
        Set to ``True`` if this force is to be mutable and optimized.
        Set to ``False`` if this force is to be held constant while
        other forces are optimized.
    nbins : int, otional
        This must be a positive integer if this force is being optimized.
        nbins is used for setting the potenials independent varible (x) range
        and step size (dx).
        It is also used in determining the bin size of the target and query
        distributions.
    correction_fit_window: int, optional
        The window size (number of data points) to use when fitting
        the iterative potential to head and tail correction forms.
        This is only used when the Force is set to be optimized.
    correction_form: Callable, default=msibi.utils.corrections.harmonic
        The type of correciton form to apply to the potential.
        This is only used when the Force is set to be optimized.
        This correction is applied to both the head and tail of the potential.
    """

    def __init__(
        self,
        type1: str,
        type2: str,
        type3: str,
        optimize: bool,
        nbins: Optional[int] = None,
        smoothing_window: Optional[int] = None,
        smoothing_order: Optional[int] = None,
        correction_fit_window: Optional[int] = None,
        correction_form: Callable = harmonic,
    ):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        name = f"{self.type1}-{self.type2}-{self.type3}"
        super(Angle, self).__init__(
            name=name,
            optimize=optimize,
            nbins=nbins,
            smoothing_window=smoothing_window,
            smoothing_order=smoothing_order,
            correction_fit_window=correction_fit_window,
            correction_form=correction_form,
        )
        if self.optimize:
            if self.smoothing_window is None:
                self.smoothing_window = 15
            if self.smoothing_order is None:
                self.smoothing_order = 3
            if self.correction_fit_window is None:
                self.correciton_fit_window = 10

    def set_harmonic(self, t0: Union[float, int], k: Union[float, int]) -> None:
        """Set a fixed harmonic angle potential.

        .. warning::

            Using this method is not compatible with :class:`Force`
            objects that are set to be optimized during MSIBI.

        .. note::

            For more information on harmonic angle potentials, refer to the
            `HOOMD-blue harmonic angle documentation <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/angle/harmonic.html>`_.

        Parameters
        ----------
        t0 : (Union[float, int])
            Equilibrium bond angle [radians]
        k : (Union[float, int])
            Spring constant [energy]

        """
        if self.optimize:
            raise RuntimeError(
                f"Force {self} is set to be optimized during MSIBI."
                "This potential setter cannot be used "
                "for a force designated for optimization. "
                "Instead, use `set_from_file` or `set_polynomial`."
            )
        self.format = "static"
        self.force_init = "Harmonic"
        self.force_entry = dict(t0=t0, k=k)

    def _table_entry(self) -> dict:
        """Set the correct entry to use in ``hoomd.md.angle.Table``"""
        table_entry = {"U": self.potential, "tau": self.force}
        return table_entry

    def _get_distribution(self, state: msibi.state.State, gsd_file: str) -> np.ndarray:
        """Calculate a bond angle distribution.

        Parameters
        ----------
        state: msibi.state.State
            State used in calculating the distribution.
        gsd_file: str
            Path to the GSD file used.
        """
        return angle_distribution(
            gsd_file=gsd_file,
            A_name=self.type1,
            B_name=self.type2,
            C_name=self.type3,
            start=-state.n_frames,
            stop=-1,
            stride=state.sampling_stride,
            histogram=True,
            normalize=True,
            as_probability=True,
            theta_min=self.x_min,
            theta_max=self.x_max,
            bins=self.nbins + 1,
        )


class Pair(Force):
    """Creates a non-bonded pair :class:`Force` used in query simulations.

    .. note::

        The pair type is sorted so that ``type1`` and ``type2``
        are listed in alphabetical order, and must match the pair
        types found in the state's target GSD file bond types.

        For example: ``Pair(type1="B", type2="A")`` will have ``Pair.name = "A-B"``

    Parameters
    ----------
    type1 : str
        Name of the first particle type in the pair.
        This must match the types found in the state's target GSD file.
    type2 : str
        Name of the second particle type in the pair.
        This must match the types found in the state's target GSD file.
    optimize : bool
        Set to ``True`` if this force is to be mutable and optimized.
        Set to ``False`` if this force is to be held constant while
        other forces are optimized.
    r_cut : (Union[float, int])
        Sets the cutoff distance used in Hoomd's neighborlist.
        This is also the maximum radius value used in the pair table potential.
    r_switch : (Union[float, int])
        Sets the radius value where the potential is corrected to
        begin approach zero smoothly.
    nbins : int, optional
        This must be a positive integer if this force is being optimized.
        nbins is used for setting the potenials independent varible (x) range
        and step size (dx).
        It is also used in determining the bin size of the target and query
        distributions.
    exclude_bonded : bool
        If ``True``, then particles from the same molecule are not
        included in the RDF calculation.
        If ``False``, all particles are included.
    correction_fit_window: int, optional
        The window size (number of data points) to use when fitting
        the iterative potential to head and tail correction forms.
        This is only used when the Force is set to be optimized.
    correction_form: Callable, default=msibi.utils.corrections.harmonic
        The type of correciton form to apply to the potential.
        This is only used when the Force is set to be optimized.
        This correction is only applied to both the head of the potential.
    """

    def __init__(
        self,
        type1: str,
        type2: str,
        optimize: bool,
        nbins: Optional[int] = None,
        r_cut: Optional[Union[float, int]] = None,
        r_switch: Optional[Union[float, int]] = None,
        correction_fit_window: Optional[int] = None,
        exclude_bonded: bool = False,
        head_correction_form: Callable = exponential,
    ):
        self.type1, self.type2 = sorted([type1, type2], key=natural_sort)
        self.r_cut = r_cut
        self.r_switch = r_switch
        name = f"{self.type1}-{self.type2}"
        # Pair types in hoomd have a different tuple naming structure.
        # Using different attr above to keep consistent msibi.force.Force.name format.
        self._pair_name = (self.type1, self.type2)
        super(Pair, self).__init__(
            name=name,
            optimize=optimize,
            nbins=nbins,
            correction_fit_window=correction_fit_window,
            correction_form=head_correction_form,
        )
        if self.optimize:
            if self.smoothing_window is None:
                self.smoothing_window = 15
            if self.smoothing_order is None:
                self.smoothing_order = 4
            if self.correction_fit_window is None:
                self.correction_fit_window = 8

    def set_lj(
        self,
        r_min: Union[float, int],
        r_cut: Union[float, int],
        epsilon: Union[float, int],
        sigma: Union[float, int],
    ) -> None:
        """Set a 12-6 Lennard Jones table pair potential used in query simulations.

        .. note::

            This creates a table potential from the LJ 12-6 function with the
            given parameters. Use this to create an initial guess when optimizing
            a :class:`Pair` force. It can still be used to set a static potential.

        Parameters
        ----------
        epsilon : (Union[float, int])
            Sets the dept hof the potential energy well.
        sigma : (Union[float, int])
            Sets the particle size.
        r_cut : (Union[float, int])
            Maximum distance used to calculate neighbors.
        """
        self.format = "table"
        self.dx = (r_cut - r_min) / self.nbins
        self.x_range = np.arange(r_min, r_cut + self.dx, self.dx)
        self.x_min = self.x_range[0]
        self.r_cut = self.x_range[-1]
        self.potential = lennard_jones(r=self.x_range, epsilon=epsilon, sigma=sigma)
        self.force_init = "Table"

    def _table_entry(self) -> dict:
        """Set the correct entry to use in ``hoomd.md.pair.Table``"""
        table_entry = {
            "r_min": self.x_min,
            "U": self.potential,
            "F": self.force,
        }
        return table_entry

    def _get_distribution(self, state: msibi.state.State, gsd_file: str) -> np.ndarray:
        """Calculate a pair distribution (RDF).

        Parameters
        ----------
        state: msibi.state.State
            State point used in calculating the distribution.
        gsd_file: str
            Path to the GSD file used.
        """
        rdf, N = gsd_rdf(
            gsdfile=gsd_file,
            A_name=self.type1,
            B_name=self.type2,
            r_min=self.x_min,
            r_max=self.r_cut,
            exclude_bonded=state.exclude_bonded,
            start=-state.n_frames,
            stride=state.sampling_stride,
            stop=-1,
            bins=self.nbins + 1,
        )
        x = rdf.bin_centers
        y = rdf.rdf
        dist = np.vstack([x, y])
        return dist.T


class Dihedral(Force):
    """Creates a dihedral :class:`Force` used in query simulations.

    .. note::

        The dihedral type is formed in the order of ``type1-type2-type3-type4`` and
        must match the same order in the target GSD file dihedral types.

    Parameters
    ----------
    type1 : str
        Name of the first particle type in the dihedral.
        This must match the types found in the state's target GSD file.
    type2 : str
        Name of the second particle type in the dihedral.
        This must match the types found in the state's target GSD file.
    type3 : str
        Name of the third particle type in the dihedral.
        This must match the types found in the state's target GSD file.
    type4 : str
        Name of the fourth particle type in the dihedral.
        This must match the types found in the state's target GSD file.
    optimize : bool
        Set to `True` if this force is to be mutable and optimized.
        Set to `False` if this force is to be held constant while
        other forces are optimized.
    nbins : int, optional
        This must be a positive integer if this force is being optimized.
        nbins is used for setting the potenials independent varible (x) range
        and step size (dx).
        It is also used in determining the bin size of the target and query
        distributions.
    correction_fit_window: int, optional
        The window size (number of data points) to use when fitting
        the iterative potential to head and tail correction forms.
        This is only used when the Force is set to be optimized.
    correction_form: Callable, default=msibi.utils.corrections.harmonic
        The type of correciton form to apply to the potential.
        This is only used when the Force is set to be optimized.
        This correction is applied to both the head and tail of the potential.
    """

    def __init__(
        self,
        type1: str,
        type2: str,
        type3: str,
        type4: str,
        optimize: bool,
        nbins: Optional[int] = None,
        smoothing_window: Optional[int] = None,
        smoothing_order: Optional[int] = None,
        correction_fit_window: Optional[int] = None,
        correction_form: Callable = harmonic,
    ):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.type4 = type4
        name = f"{self.type1}-{self.type2}-{self.type3}-{self.type4}"
        super(Dihedral, self).__init__(
            name=name,
            optimize=optimize,
            nbins=nbins,
            smoothing_window=smoothing_window,
            smoothing_order=smoothing_order,
            correction_fit_window=correction_fit_window,
            correction_form=correction_form,
        )
        if self.optimize:
            if self.smoothing_window is None:
                self.smoothing_window = 15
            if self.smoothing_order is None:
                self.smoothing_order = 2
            if self.correction_fit_window is None:
                self.correciton_fit_window = 10

    def set_periodic(
        self,
        phi0: Union[float, int],
        k: Union[float, int],
        d: int,
        n: int,
    ) -> None:
        """Set a fixed periodic dihedral potential.

        .. warning::

            Using this method is not compatible with :class:`Force`
            objects that are set to be optimized during MSIBI.

        .. note::

            For more information on periodic dihedral potentials, refer to the
            `HOOMD-blue dihedral periodic documentation <https://hoomd-blue.readthedocs.io/en/latest/hoomd/md/dihedral/periodic.html>`_.

        Parameters
        ----------
        phi0 : float
            Phase shift [radians]
        k : float
            Spring constant [energy]
        d : int
            Sign factor
        n : int
            Angle multiplicity
        """
        if self.optimize:
            raise RuntimeError(
                f"Force {self} is set to be optimized during MSIBI."
                "This potential setter cannot be used "
                "for a force designated for optimization. "
                "Instead, use `set_from_file` or `set_polynomial`."
            )
        self.format = "static"
        self.force_init = "Periodic"
        self.force_entry = dict(phi0=phi0, k=k, d=d, n=n)

    def _table_entry(self) -> dict:
        """Set the correct entry to use in ``hoomd.md.dihedral.Table``"""
        table_entry = {"U": self.potential, "tau": self.force}
        return table_entry

    def _get_distribution(self, state: msibi.state.State, gsd_file: str) -> np.ndarray:
        """Calculate a dihedral angle distribution.

        Parameters
        ----------
        state: msibi.state.State
            State used in calculating the distribution.
        gsd_file: str
            Path to the GSD file used.
        """
        return dihedral_distribution(
            gsd_file=gsd_file,
            A_name=self.type1,
            B_name=self.type2,
            C_name=self.type3,
            D_name=self.type4,
            start=-state.n_frames,
            stop=-1,
            stride=state.sampling_stride,
            histogram=True,
            normalize=True,
            as_probability=True,
            bins=self.nbins + 1,
        )
