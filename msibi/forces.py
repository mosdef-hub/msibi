import math
import os
from typing import Union
import warnings

from cmeutils.structure import (
        angle_distribution,
        bond_distribution,
        dihedral_distribution,
        gsd_rdf
)
import matplotlib.pyplot as plt
import numpy as np

import msibi
from msibi.potentials import quadratic_spring, bond_correction
from msibi.utils.error_calculation import calc_similarity
from msibi.utils.smoothing import savitzky_golay
from msibi.utils.sorting import natural_sort


class Force(object):
    """
    Base class from which other forces inherit.
    Don't call this class directly, instead use
    msibi.forces.Bond, msibi.forces.Angle, msibi.forces.Pair,
    and msibi.forces.Dihedral.

    Forces in MSIBI can either be held constant (i.e. fixed) or
    optimized (i.e. mutable). Only one type of of force
    can be optimized at a time (i.e. angles, or pairs, etc..)

    Parameters
    ----------
    name : str, required
        The name of the type in the Force.
        Must match the names found in the State's .gsd trajectory file.
    optimize : bool, required
        Set to True if this force is to be mutable and optimized.
        Set to False if this force is to be held constant while
        other forces are optimized.
    nbins : int, optional
        This must be a positive integer if this force is being optimized.
        nbins is used to setting the potenials independent varible (x) range
        and step size (dx).
        It is also used in determining the bin size of the target and query
        distributions.

    """

    def __init__(
            self,
            name: str,
            optimize: bool,
            nbins: int=None,
            head_correction_form: str="linear"
    ):
        if optimize and nbins is None or nbins<=0:
            raise ValueError(
                    "If a force is set to be optimized, nbins must be "
                    "a positive, non-zero integer."
            )
        self.name = name
        self.optimize = optimize
        self.head_correction_form = head_correction_form
        self.format = None
        self.xmin = None
        self.xmax = None
        self.dx = None
        self.x_range = None
        self.potential_history = []
        self._potential = None
        self._smoothing_window = 3
        self._smoothing_order = 1
        self._nbins = nbins
        self._states = dict()
        self._head_correction_history = []
        self._tail_correction_history = []
        self._learned_potential_history = []

        if optimize and nbins is None:
            raise ValueError(
                    "If this force is set to be optimized, the nbins "
                    "must be set as a non-zero value"
            )

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
        return self._potential

    @potential.setter
    def potential(self, array):
        if self.format != "table":
            #TODO: Make custom error for this
            raise ValueError(
                    "Setting potential arrays can only be done "
                    "for Forces that utilize tables. "
                    "See msibi.forces.Force.set_quadratic() or "
                    "msibi.forces.Force.set_from_file()"
            )
        self._potential = array

    @property
    def force(self) -> np.ndarray:
        """The force values F(x)."""
        if self.format != "table":
            warnings.warn(f"{self} is not using a table potential.")
            return None
        return -1.0*np.gradient(self.potential, self.dx)

    @property
    def smoothing_window(self) -> int:
        """Window size used in smoothing the distributions."""
        return self._smoothing_window

    @smoothing_window.setter
    def smoothing_window(self, value: int):
        if not isinstance(value, int):
            raise ValueError("The smoothing window must be an integer.")
        self._smoothing_window = value
        for state in self._states:
            self._add_state(state)

    @property
    def smoothing_order(self) -> int:
        """The order used in smoothing the distributions."""
        return self._smoothing_order

    @smoothing_order.setter
    def smoothing_order(self, value: int):
        if not isinstance(value, int):
            raise ValueError("The smoothing order must be an integer.")
        self._smoothing_order = value
        for state in self._states:
            self._add_state(state)

    @property
    def nbins(self) -> int:
        """The number of bins used in calculating distributions."""
        return self._nbins

    @nbins.setter
    def nbins(self, value: int):
        if not isinstance(value, int):
            raise ValueError("nbins must be an integer.")
        self._nbins =  value
        for state in self._states:
            self._add_state(state)

    def target_distribution(self, state: msibi.state.State) -> np.ndarray:
        """The target structural distribution corresponding to this foce.

        Parameters
        ----------
        state : msibi.state.State, required
            The state to use in finding the target distribution.

        """
        return self._states[state]["target_distribution"]

    def plot_target_distribution(self, state: msibi.state.State) -> None:
        """
        Quick plotting function that shows the target structural
        distribution corresponding to this forces.

        Parameters
        ----------
        state : msibi.state.State, required
            The state to use in finding the target distribution.

        Notes
        -----
        Use this to see how the shape of the target distribution is
        affected by your choices for nbins, smoothing window,
        and smoothing order.

        """
        #TODO: Make custom error
        if not self.optimize:
            raise RuntimeError(
                    "This force object is not set to be optimized. "
                    "The target distribution is not calculated."
            )
        target = self.target_distribution(state)
        fig = plt.figure()
        plt.title(f"State {state.name}: {self.name} Target")
        plt.ylabel("P(x)")
        plt.xlabel("x")
        plt.plot(target[:,0], target[:,1])
        if self.smoothing_window:
            y_smoothed = savitzky_golay(
                    target[:,1],
                    window_size=self.smoothing_window,
                    order=self.smoothing_order
            )
            plt.plot(target[:,0], y_smoothed, label="Smoothed")
            plt.legend()

    def plot_fit_scores(self, state: msibi.state.State) -> None:
       """Plots the evolution of the distribution
       matching evolution.

        Parameters
        ----------
        state : msibi.state.State, required
            The state to use in finding the target distribution.

       """
       if not self.optimize:
            raise RuntimeError("This force object is not set to be optimized.")
       fig = plt.figure()
       plt.plot(self._states[state]["f_fit"], "o-")
       plt.xlabel("Iteration")
       plt.ylabel("Fit Score")

    def distribution_history(self, state: msibi.state.State):
        """Get the complete query distribution history for a given state.

        Parameters
        ----------
        state : msibi.state.State, required
            The state to use for calculating the distribution.

        """
        return self._states[state]["distribution_history"]

    def set_target_distribution(self, state: msibi.state.State, array) -> None:
        """Store the target distribution for a given state.

        Parameters
        ----------
        state: msibi.state.State
            The state used in finding the distribution.

        """
        self._states[state]["target_distribution"] = array

    def current_distribution(
            self,
            state: msibi.state.State,
            query: bool=True
    ) -> np.ndarray:
        """Returns the corresponding distrubution from the most recent
        query simulation.

        Parameters
        ----------
        state : msibi.state.State, required
            The state to use for calculating the distribution.

        """
        return self._get_state_distribution(state, query)

    def distribution_fit(self, state: msibi.state.State) -> float:
        """Get the fit score from the most recent query simulation.

        Parameters
        ----------
        state : msibi.state.State, required
            The state to use for calculating the distribution.

        """
        return self._calc_fit(state)

    def set_quadratic(
            self,
            k4: Union[float, int],
            k3: Union[float, int],
            k2: Union[float, int],
            x0: Union[float, int],
            x_min: Union[float, int],
            x_max: Union[float, int]
    ) -> None:
        """Set a potential based on the following function:

            V(x) = k4(x-x0)^4 + k3(x-x0)^3 + k2(x-x0)^2

        Using this method will create a table potential V(x) over the range
        x_min - x_max.

        This should be the potential form of choice when setting an initial
        guess potential for the force to be optimized.

        Parameters
        ----------
        x0, k4, k3, k2 : float, required
            The paraters used in the V(x) function described above
        x_min : float, required
            The lower bound of the potential range
        x_max : float, required
            The upper bound of the potential range

        """
        self.format = "table"
        self.x_min = x_min
        self.x_max = x_max
        self.dx = x_max / self.nbins
        self.x_range = np.arange(x_min, x_max + self.dx, self.dx)
        self.potential = quadratic_spring(self.x_range, x0, k4, k3, k2)
        self.force_init = "Table"
        self.force_entry = self._table_entry()

    def set_from_file(self, file_path: str) -> None:
        """Set a potential from a text file.

        Parameters:
        -----------
        file_path : str, required
            The full path to the table potential text file.

        Notes:
        ------
        The columns of the text file must be in the order of r, V.
        where r is the independent value (i.e. distance) and V
        is the potential enregy at r. The force will be calculated
        from r and V using np.gradient().

        Use this potential setter to set a potential from a previous MSIBI run.
        For example, use the final potential files from a bond-optimization IBI
        run to set a static coarse-grained bond potential while you perform
        IBI runs on angle and/or pair potentials.

        """
        f = np.loadtxt(file_path)
        self.x_range = f[:,0]
        self.dx = np.round(self.x_range[1] - self.x_range[0], 3)
        self.x_min = self.x_range[0]
        self.x_max = self.x_range[-1] + self.dx
        self._potential = f[:,1]
        self.format = "table"
        self.force_init = "Table"

    def _add_state(self, state: msibi.state.State) -> None:
        """Add a state to be used in optimizing this force.

        Parameters
        ----------
        state : msibi.state.State
            Instance of a State object previously created.

        """
        if self.optimize:
            target_distribution = self._get_state_distribution(
                    state=state, query=False
            )
            if self.smoothing_window and self.smoothing_order:
                target_distribution[:,1] = savitzky_golay(
                        y=target_distribution[:,1],
                        window_size=self.smoothing_window,
                        order=self.smoothing_order,
                        deriv=0,
                        rate=1
                )

        else:
            target_distribution = None
        self._states[state] = {
                "target_distribution": target_distribution,
                "current_distribution": None,
                "alpha": state.alpha,
                "f_fit": [],
                "distribution_history": [],
                "path": state.dir
        }

    def _compute_current_distribution(self, state: msibi.state.State) -> None:
        """Find the current distribution of the query trajectory

        Parameters
        ----------
        state : msibi.state.State
            Instance of a State object previously created.

        """
        distribution = self._get_state_distribution(state, query=True)
        if self.smoothing_window and self.smoothing_order:
            distribution[:,1] = savitzky_golay(
                    y=distribution[:,1],
                    window_size=self.smoothing_window,
                    order=self.smoothing_order,
                    deriv=0,
                    rate=1
            )
            negative_idx = np.where(distribution[:,1] < 0)[0]
            distribution[:,1][negative_idx] = 0
        self._states[state]["current_distribution"] = distribution

        f_fit = calc_similarity(
                    distribution[:,1],
                    self._states[state]["target_distribution"][:,1]
        )
        self._states[state]["f_fit"].append(f_fit)

    def _get_state_distribution(
            self,
            state: msibi.state.State,
            query: bool
    ) -> np.ndarray:
        """Get the corresponding distrubiton for a given state.

        Parameters
        ----------
        state: msibi.state.State
            State used in calculating the distribution.
        query: bool
            If True, uses the most recent query trajectory.
            If False, uses the state's target trajectory.

        """
        if query:
            traj = state.query_traj
        else:
            traj = state.traj_file
        return self._get_distribution(state=state, gsd_file=traj)

    def _save_current_distribution(
            self,
            state: msibi.state.State,
            iteration: int
    ) -> None:
        """Save the corresponding distrubiton for a given state to a file.

        Parameters
        ----------
        state : State
            The state used in finding the distribution.
        iteration : int
            Current iteration step, used in the filename.

        """
        distribution = self._states[state]["current_distribution"]
        distribution[:,0] -= self.dx / 2
        fname = f"dist_{self.name}-state_{state.name}-step_{iteration}.txt"
        fpath = os.path.join(state.dir, fname)
        np.savetxt(fpath, distribution)

    def _update_potential(self) -> None:
        """Compare distributions of current iteration against target,
        and update the potential via Boltzmann inversion.

        """
        self.potential_history.append(np.copy(self.potential))
        for state in self._states:
            kT = state.kT
            current_dist = self._states[state]["current_distribution"]
            target_dist = self._states[state]["target_distribution"]
            self._states[state]["distribution_history"].append(current_dist)
            N = len(self._states)
            #TODO: Use potential setter here? Does it work with +=?
            self._potential += state.alpha * (
                    kT * np.log(current_dist[:,1] / target_dist[:,1]) / N
            )
        #TODO: Add correction funcs to Force classes
        #TODO: Smoothing potential before doing head and tail corrections?
        self._potential, real, head_cut, tail_cut = self._correction_function(
                self.x_range, self.potential, self.head_correction_form
        )
        self._head_correction_history.append(np.copy(self.potential[0:head_cut]))
        self._tail_correction_history.append(np.copy(self.potential[tail_cut:]))
        self._learned_potential_history.append(np.copy(self.potential[real]))


class Bond(Force):
    """
    Creates a bond force used in query simulations.

    Parameters
    ----------
    type1 : str, required
        Name of the first particle type in the bond.
        This must match the types found in the state's target GSD file.
    type2 : str, required
        Name of the second particle type in the bond.
        This must match the types found in the state's target GSD file.
    optimize : bool, required
        Set to True if this force is to be mutable and optimized.
        Set to False if this force is to be held constant while
        other forces are optimized.
    nbins : int, optional
        This must be a positive integer if this force is being optimized.
        nbins is used to setting the potenials independent varible (x) range
        and step size (dx).
        It is also used in determining the bin size of the target and query
        distributions.

    Notes
    -----
    The bond type is sorted so that type1 and type2
    are listed in alphabetical order, and must match the bond
    types found in the state's target GSD file bond types.

    For example: `Bond(type1="B", type2="A")` will have `Bond.name = "A-B"`

    """

    def __init__(
            self,
            type1: str,
            type2: str,
            optimize: bool,
            nbins: int=None,
            head_correction_form: str="linear"
    ):
        self.type1, self.type2 = sorted(
                    [type1, type2], key=natural_sort
        )
        self._correction_function = bond_correction
        name = f"{self.type1}-{self.type2}"
        super(Bond, self).__init__(
                name=name,
                optimize=optimize,
                nbins=nbins,
                head_correction_form=head_correction_form
        )

    def set_harmonic(self, r0: Union[float, int], k: Union[float, int]) -> None:
        """Set a fixed harmonic bond potential.
        Using this method is not compatible force msibi.forces.Force
        objects that are set to be optimized during MSIBI

        Parameters
        ----------
        r0 : (Union[float, int]), required
            Equilibrium bond length
        k : (Union[float, int]), required
            Spring constant

        """
        if self.optimize:
            raise RuntimeError(
                    f"Force {self} is set to be optimized during MSIBI."
                    "This potential setter cannot be used "
                    "for a force designated for optimization. "
                    "Instead, use set_from_file() or set_quadratic()."
            )
        self.type = "static"
        self.force_init = "Harmonic"
        self.force_entry = dict(r0=r0, k=k)

    def _table_entry(self) -> dict:
        table_entry = {
                "r_min": self.x_min,
                "r_max": self.x_max,
                "U": self.potential,
                "F": self.force
        }
        return table_entry

    def _get_distribution(
            self,
            state: msibi.state.State,
            gsd_file: str
    ) -> np.ndarray:
        """Calculate a bond length distribution.

        Parameters
        ----------
        state: msibi.state.State, required
            State used in calculating the distribution.
        gsd_file: str, required
            Path to the GSD file used.

        """
        return bond_distribution(
                gsd_file=gsd_file,
                A_name=self.type1,
                B_name=self.type2,
                start=-state.n_frames,
                histogram=True,
                normalize=True,
                l_min=self.x_min,
                l_max=self.x_max,
                bins=self.nbins + 1
        )


class Angle(Force):
    """
    Creates an angle force used in query simulations.

    Parameters
    ----------
    type1 : str, required
        Name of the first particle type in the angle.
        This must match the types found in the state's target GSD file.
    type2 : str, required
        Name of the second particle type in the angle.
        This must match the types found in the state's target GSD file.
    type3 : str, required
        Name of the third particle type in the angle.
        This must match the types found in the state's target GSD file.
    optimize : bool, required
        Set to True if this force is to be mutable and optimized.
        Set to False if this force is to be held constant while
        other forces are optimized.
    nbins : int, optional
        This must be a positive integer if this force is being optimized.
        nbins is used to setting the potenials independent varible (x) range
        and step size (dx).
        It is also used in determining the bin size of the target and query
        distributions.

    Notes
    -----
    The angle type is formed in the order of type1-type2-type3 and must match
    the same order in the target GSD file angle types.

    """

    def __init__(
            self,
            type1: str,
            type2: str,
            type3: str,
            optimize: bool,
            nbins: int=None,
            head_correction_form: str="linear"
    ):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        name = f"{self.type1}-{self.type2}-{self.type3}"
        self._correction_function = bond_correction
        super(Angle, self).__init__(
                name=name,
                optimize=optimize,
                nbins=nbins,
                head_correction_form=head_correction_form
        )

    def set_harmonic(self, t0: Union[float, int], k: Union[float, int]) -> None:
        """Set a fixed harmonic angle potential.
        Using this method is not compatible force msibi.forces.Force
        objects that are set to be optimized during MSIBI

        Parameters
        ----------
        t0 : (Union[float, int]), required
            Equilibrium bond angle
        k : (Union[float, int]), required
            Spring constant

        """
        if self.optimize:
            raise RuntimeError(
                    f"Force {self} is set to be optimized during MSIBI."
                    "This potential setter cannot be used "
                    "for a force designated for optimization. "
                    "Instead, use set_from_file() or set_quadratic()."
            )
        self.type = "static"
        self.force_init = "Harmonic"
        self.force_entry = dict(t0=t0, k=k)

    def _table_entry(self) -> dict:
        table_entry = {"U": self.potential, "tau": self.force}
        return table_entry

    def _get_distribution(
            self,
            state: msibi.state.State,
            gsd_file: str
    ) -> np.ndarray:
        """Calculate a bond angle distribution.

        Parameters
        ----------
        state: msibi.state.State, required
            State used in calculating the distribution.
        gsd_file: str, required
            Path to the GSD file used.

        """
        return angle_distribution(
                gsd_file=gsd_file,
                A_name=self.type1,
                B_name=self.type2,
                C_name=self.type3,
                start=-state.n_frames,
                histogram=True,
                normalize=True,
                theta_min=self.x_min,
                theta_max=self.x_max,
                bins=self.nbins + 1
        )


class Pair(Force):
    """
    Creates a pair force used in query simulations.

    Parameters
    ----------
    type1 : str, required
        Name of the first particle type in the pair.
        This must match the types found in the state's target GSD file.
    type2 : str, required
        Name of the second particle type in the pair.
        This must match the types found in the state's target GSD file.
    optimize : bool, required
        Set to True if this force is to be mutable and optimized.
        Set to False if this force is to be held constant while
        other forces are optimized.
    r_cut : (Union[float, int]) : required
        Sets the cutoff distance used in Hoomd's neighborlist.
    nbins : int, optional
        This must be a positive integer if this force is being optimized.
        nbins is used to setting the potenials independent varible (x) range
        and step size (dx).
        It is also used in determining the bin size of the target and query
        distributions.
    exclude_bonded : bool, default False
        If True, then particles from the same molecule are not
        included in the RDF calculation.
        If False, all particles are included.

    Notes
    -----
    The pair type is sorted so that type1 and type2
    are listed in alphabetical order, and must match the pair
    types found in the state's target GSD file bond types.

    For example: `Pair(type1="B", type2="A")` will have `Pair.name = "A-B"`

    """

    def __init__(
            self,
            type1: str,
            type2: str,
            optimize: bool,
            r_cut: Union[float, int],
            nbins: int=None,
            exclude_bonded: bool=False,
            head_correction_form: str="linear"
    ):
        self.type1, self.type2 = sorted( [type1, type2], key=natural_sort)
        self.r_cut = r_cut
        name = f"{self.type1}-{self.type2}"
        # Pair types in hoomd have different naming structure.
        self._pair_name = (type1, type2)
        super(Pair, self).__init__(
                name=name,
                optimize=optimize,
                nbins=nbins,
                head_correction_form=head_correction_form
        )

    def set_lj(
            self,
            epsilon: Union[float, int],
            sigma: Union[float, int]
    ) -> None:
        """Set a hoomd 12-6 LJ pair potential used during
        the query simulations.

        Parameters
        ----------
        epsilon : (Union[float, int]), required
            Sets the dept hof the potential energy well.
        sigma : (Union[float, int]), required
            Sets the particle size.
        r_cut : (Union[float, int]), required
            Maximum distance used to calculate neighbor pair potentials.

        """
        if self.optimize:
            raise RuntimeError(
                    f"Force {self} is set to be optimized during MSIBI."
                    "This potential setter cannot be used "
                    "for a force designated for optimization. "
                    "Instead, use set_from_file() or set_quadratic()."
            )
        self.type = "static"
        self.force_init = "LJ"
        self.force_entry = dict(sigma=sigma, epsilon=epsilon, r_cut=self.r_cut)

    def _table_entry(self) -> dict:
        table_entry = {
                "r_min": self.x_min,
                "U": self.potential,
                "F": self.force,
                "r_cut": self.r_cut
        }
        return table_entry

    def _get_distribution(
            self,
            state: msibi.state.State,
            gsd_file: str
    ) -> np.ndarray:
        """Calculate a pair distribution.

        Parameters
        ----------
        state: msibi.state.State, required
            State used in calculating the distribution.
        gsd_file: str, required
            Path to the GSD file used.

        """
        return gsd_rdf(
                gsdfile=gsd_file,
                A_name=self.type1,
                B_name=self.type2,
                start=-state.n_frames,
                stop=-1,
                bins=self.nbins + 1
        )


class Dihedral(Force):
    """
    Creates a dihedral force used in query simulations.

    Parameters
    ----------
    type1 : str, required
        Name of the first particle type in the dihedral.
        This must match the types found in the state's target GSD file.
    type2 : str, required
        Name of the second particle type in the dihedral.
        This must match the types found in the state's target GSD file.
    type3 : str, required
        Name of the third particle type in the dihedral.
        This must match the types found in the state's target GSD file.
    type4 : str, required
        Name of the fourth particle type in the dihedral.
        This must match the types found in the state's target GSD file.
    optimize : bool, required
        Set to True if this force is to be mutable and optimized.
        Set to False if this force is to be held constant while
        other forces are optimized.
    nbins : int, optional
        This must be a positive integer if this force is being optimized.
        nbins is used to setting the potenials independent varible (x) range
        and step size (dx).
        It is also used in determining the bin size of the target and query
        distributions.

    Notes
    -----
    The dihedral type is formed in the order of type1-type2-type3-type4 and
    must match the same order in the target GSD file angle types.

    """

    def __init__(
            self,
            type1: str,
            type2: str,
            type3: str,
            type4: str,
            optimize: bool,
            nbins: int=None,
            head_correction_form: str="linear"
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
                head_correction_form=head_correction_form
        )

    def set_harmonic(
            self,
            phi0: Union[float, int],
            k: Union[float, int],
            d: int,
            n: int,
    ) -> None:
        """Set a fixed harmonic dihedral potential.

        Parameters
        ----------
        phi0 : float, required
            Equilibrium bond length
        k : float, required
            Spring constant
        d : int, required
            Sign factor
        n : int, required
            Angle scaling factor

        """
        if self.optimize:
            raise RuntimeError(
                    f"Force {self} is set to be optimized during MSIBI."
                    "This potential setter cannot be used "
                    "for a force designated for optimization. "
                    "Instead, use set_from_file() or set_quadratic()."
            )
        self.type = "static"
        self.force_init = "Periodic"
        self.force_entry = dict(phi0=phi0, k=k, d=d, n=n)

    def _table_entry(self) -> dict:
        table_entry = {"U": self.potential, "tau": self.force}
        return table_entry

    def _get_distribution(
            self,
            state: msibi.state.State,
            gsd_file: str
    ) -> np.ndarray:
        """Calculate a dihedral distribution.

        Parameters
        ----------
        state: msibi.state.State, required
            State used in calculating the distribution.
        gsd_file: str, required
            Path to the GSD file used.

        """
        return dihedral_distribution(
                gsd_file=gsd,
                A_name=self.type1,
                B_name=self.type2,
                C_name=self.type3,
                D_name=self.type4,
                start=-state.n_frames,
                histogram=True,
                normalize=True,
                bins=self.nbins + 1
        )
