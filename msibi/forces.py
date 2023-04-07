import math
import os

import matplotlib.pyplot as plt
import numpy as np

from cmeutils.structure import (
        angle_distribution, bond_distribution, dihedral_distribution, gsd_rdf
)
from msibi.potentials import quadratic_spring, bond_correction 
from msibi.utils.error_calculation import calc_similarity
from msibi.utils.smoothing import savitzky_golay
from msibi.utils.sorting import natural_sort


HARMONIC_BOND_ENTRY = "harmonic_bond.bond_coeff.set('{}', k={}, r0={})"
TABLE_BOND_ENTRY = "btable.set_from_file('{}', '{}')"
HARMONIC_ANGLE_ENTRY = "harmonic_angle.angle_coeff.set('{}', k={}, t0={})"
TABLE_ANGLE_ENTRY = "atable.set_from_file('{}', '{}')"
LJ_PAIR_ENTRY = "lj.pair_coeff.set('{}', '{}', epsilon={}, sigma={}, r_cut={})"
TABLE_PAIR_ENTRY = "table.set_from_file('{}', '{}', filename='{}')"
HARMONIC_DIHEDRAL_ENTRY = "harmonic_dihedral.dihedral_coeff.set('{}', k={}, d={}, n={}, phi0={})" 
TABLE_DIHEDRAL_ENTRY = "dtable.set_from_file('{}', '{}')" 


class Force(object):
    """Creates a potential, either to be held constant, or to be
    optimized.

    Parameters
    ----------
    name : str, required
        The name of the type in the bond.
        Must match the names found in the State's .gsd trajectory file

    """
    def __init__(self, name, optimize=False, head_correction_form="linear"):
        self.name = name 
        self.optimize = optimize
        self._potential_file = "{}" 
        self.potential = None 
        self.head_correction_form = head_correction_form
        self.xmin = None
        self.xmax = None
        self.dx = None
        self.x_range = None
        self._smoothing_window = 3
        self._smoothing_order = 1
        self._nbins = 100
        self.format = None
        self._states = dict()
        self.potential_history = []
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
    def smoothing_window(self):
        return self._smoothing_window

    @smoothing_window.setter
    def smoothing_window(self, value):
        self._smoothing_window = value

    @property
    def smoothing_order(self):
        return self._smoothing_order

    @smoothing_order.setter
    def smoothing_order(self, value):
        self._smoothing_order = value

    @property
    def nbins(self):
        return self._nbins

    @nbins.setter
    def nbins(self, value):
        self._nbins =  value
        for state in self._states:
            self._add_state(state)

    def target_distribution(self, state):
        return self._states[state]["target_distribution"]
   
    def plot_target_distribution(self, state):
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

    def set_target_distribution(self, state, array):
        self._states[state]["target_distribution"] = array

    def current_distribution(self, state, query=True):
        return self._get_state_distribution(state, query)

    def distribution_fit(self, state):
        return self._calc_fit(state)
    
    def set_quadratic(self, k4, k3, k2, x0, x_min, x_max, n_points=101):
        """Set a bond potential based on the following function:

            V(x) = k4(l-x0)^4 + k3(l-x0)^3 + k2(l-x0)^2

        Using this method will create a table potential V(x) over the range
        x_min - x_max.

        This should be the potential form of choice when setting an initial 
        potential for the force to be optimized.

        Parameters
        ----------
        x0, k4, k3, k2 : float, required
            The paraters used in the V(x) function described above
        x_min : float, required
            The lower bound of the bond potential lengths
        x_max : float, required
            The upper bound of the bond potential lengths
        n_points : int, default = 101 
            The number of points between l_min-l_max used to create
            the table potential

        """
        self.format = "table"
        self.x_min = x_min
        self.x_max = x_max
        self.dx = x_max / self.nbins
        self.x_range = np.arange(x_min, x_max, self.dx)
        self.potential = quadratic_spring(self.x_range, x0, k4, k3, k2)
        self.force_init = f"btable = hoomd.md.bond.table(width={self.nbins})"
        self.force_entry = TABLE_BOND_ENTRY.format(
                self.name, self._potential_file
        ) 

    def set_from_file(self, file_path):
        """Creates a bond-stretching potential from a text file.
        The columns of the text file must be in the order of r, V, F
        which is the format used by hoomd-blue for table files.

        Use this potential setter to set a potential from a previous MSIBI run.
        For example, use the final potential files from a bond-optimization IBI
        run to set a static coarse-grained bond potential while you perform
        IBI runs on angle and pair potentials.

        Parameters:
        -----------
        file_path : str, required
            The full path to the table potential text file.

        """
        self._potential_file = file_path
        f = np.loadtxt(self._potential_file)
        self.x_range = f[:,0]
        self.dx = np.round(self.x_range[1] - self.x_range[0], 3) 
        self.potential = f[:,1]
        self.x_min = self.l_range[0]
        self.x_max = self.l_range[-1] + self.dx

        self.format = "table"
        self.force_init = f"btable = hoomd.md.bond.table(width={self.nbins})"
        self.force_entry = TABLE_BOND_ENTRY.format(
                self.name, self._potential_file
        ) 

    def update_potential_file(self, fpath):
        """Set (or reset) the path to a table potential file.
        This function ensures that the Force.force_entry attribute
        is correctly updated when a potential file path is generated
        or updated.

        Parameters:
        -----------
        fpath : str, required
            Full path to the text file

        """
        self._potential_file = fpath
        self.force_entry = self.force_entry.format(self._potential_file)

    def _add_state(self, state):
        """Add a state to be used in optimizing this Fond.

        Parameters
        ----------
        state : msibi.state.State
            Instance of a State object already created.

        """
        if self.optimize:
            target_distribution = self._get_state_distribution(
                    state=state, query=False
            )
        else:
            target_distribution = None
        self._states[state] = {
                "target_distribution": target_distribution,
                "current_distribution": None,
                "alpha": state.alpha,
                "alpha_form": "linear",
                "f_fit": [],
                "path": state.dir
        }

    def _get_state_distribution(self, state, query):
        """Find the bond length distribution of a Bond at a State."""
        if query:
            traj = state.query_traj
        else:
            traj = state.traj_file
        return self._get_distribution(state=state, gsd_file=traj)

    def _compute_current_distribution(self, state):
        """Find the current distribution of the query trajectory"""
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

    def _save_current_distribution(self, state, iteration):
        """Save the current bond length distribution 

        Parameters
        ----------
        state : State
            A state object
        iteration : int
            Current iteration step, used in the filename

        """
        distribution = self._states[state]["current_distribution"]
        distribution[:,0] -= self.dx / 2
        fname = f"dist_{self.name}-state_{state.name}-step_{iteration}.txt"
        fpath = os.path.join(state.dir, fname)
        np.savetxt(fpath, distribution)

    def _update_potential(self):
        """Compare distributions of current iteration against target,
        and update the Bond potential via Boltzmann inversion.

        """
        self.potential_history.append(np.copy(self.potential))
        for state in self._states:
            kT = state.kT
            current_dist = self._states[state]["current_distribution"]
            target_dist = self._states[state]["target_distribution"]
            N = len(self._states)
            self.potential += state.alpha * (
                    kT * np.log(current_dist[:,1] / target_dist[:,1]) / N
            )
        #TODO: Add correction funcs to Force classes
        #TODO: Smoothing potential before doing head and tail corrections?
        self.potential, real, head_cut, tail_cut = self._correction_function(
                self.x_range, self.potential, self.head_correction_form
        )
        self._head_correction_history.append(self.potential[0:head_cut])
        self._tail_correction_history.append(self.potential[tail_cut:])
        self._learned_potential_history.append(self.potential[real])
        

class Bond(Force):
    def __init__(self, type1, type2, optimize, head_correction_form="linear"):
        self.type1, self.type2 = sorted(
                    [type1, type2],
                    key=natural_sort
                )
        self.force_type = "bond"
        self._correction_function = bond_correction
        name = f"{self.type1}-{self.type2}"
        super(Bond, self).__init__(
                name=name,
                optimize=optimize,
                head_correction_form=head_correction_form
        )

    def set_harmonic(self, l0, k):
        """Sets a fixed harmonic bond potential.
        Using this method is not compatible force msibi.forces.Force
        objects that are set to be optimized during MSIBI

        Parameters
        ----------
        l0 : float, required
            Equilibrium bond length
        k : float, required
            Spring constant
        """
        if self.optimize:
            raise RuntimeError(
                    f"Force {self} is set to be optimized during MSIBI."
                    "This potential setter cannot be used "
                    "for a force set for optimization. Instead, use either "
                    "set_from_file() or set_quadratic()."
            )
        self.type = "static"
        self.force_init = "harmonic_bond = hoomd.md.bond.harmonic()"
        self.force_entry = HARMONIC_BOND_ENTRY.format(self.name, k, l0)

    def _get_distribution(self, state, gsd_file):
        return bond_distribution(
                gsd_file=gsd_file,
                A_name=self.type1,
                B_name=self.type2,
                start=-state.n_frames,
                histogram=True,
                normalize=True,
                l_min=self.x_min,
                l_max=self.x_max,
                bins=self.nbins
        )        


class Angle(Force):
    def __init__(
            self,
            type1,
            type2,
            type3,
            optimize,
            head_correction_form="linear"
    ):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        name = f"{self.type1}-{self.type2}-{self.type3}"
        self.force_type = "angle"
        super(Angle, self).__init__(
                name=name,
                optimize=optimize,
                head_correction_form=head_correction_form
        )

    def set_harmonic(self, theta0, k):
        """Sets a fixed harmonic angle potential.
        Using this method is not compatible force msibi.forces.Force
        objects that are set to be optimized during MSIBI

        Parameters
        ----------
        theta0 : float, required
            Equilibrium bond angle 
        k : float, required
            Spring constant
        """
        if self.optimize:
            raise RuntimeError(
                    f"Force {self} is set to be optimized during MSIBI."
                    "This potential setter cannot be used "
                    "for a force set for optimization. Instead, use either "
                    "set_from_file() or set_quadratic()."
            )
        self.type = "static"
        self.force_init = "harmonic_angle = hoomd.md.angle.harmonic()"
        self.force_entry = HARMONIC_ANGLE_ENTRY.format(self.name, k, theta0)

    def _get_distribution(self, gsd_file):
        return angle_distribution(
                gsd_file=gsd,
                A_name=self.type1,
                B_name=self.type2,
                C_name=self.type3,
                start=-state.n_frames,
                histogram=True,
                normalize=True,
                l_min=self.x_min,
                l_max=self.x_max,
                bins=self.nbins
        )        


class Pair(Force):
    def __init__(
            self,
            type1,
            type2,
            optimize,
            exclude_bonded=False,
            head_correction_form="linear"
    ):
        self.type1, self.type2 = sorted( [type1, type2], key=natural_sort)
        name = f"{self.type1}-{self.type2}"
        self.force_type = "pair"
        super(Pair, self).__init__(
                name=name,
                optimize=optimize,
                head_correction_form=head_correction_form
        )

    def set_lj(self, epsilon, sigma, r_cut):
        """Creates a hoomd 12-6 LJ pair potential used during
        the query simulations. This method is not compatible when
        optimizing pair potentials. Rather, this method should
        only be used to create static pair potentials while optimizing
        other potentials.

        Parameters
        ----------
        epsilon : float, required
            Sets the dept hof the potential energy well.
        sigma : float, required
            Sets the particle size.
        r_cut : float, required
            Maximum distance used to calculate neighbor pair potentials.

        """
        self.type = "static"
        self.force_init = f"lj = hoomd.md.pair.lj(nlist=nl, r_cut={r_cut})"
        self.force_entry = LJ_PAIR_ENTRY.format(
                self.type1, self.type2, epsilon, sigma, r_cut
        )

    def _get_distribution(self, state, gsd_file):
        return gsd_rdf(
                gsdfile=gsd_file,
                A_name=self.type1,
                B_name=self.type2,
                start=-state.n_frames,
                stop=-1,
                bins=self.nbins
        )        


class Dihedral(Force):
    def __init__(
            self,
            type1,
            type2,
            type3,
            type4,
            optimize,
            head_correction_form="linear"
    ):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.type4 = type4
        name = f"{self.type1}-{self.type2}-{self.type3}-{self.type4}"
        self.force_type = "dihedral"
        super(Dihedral, self).__init__(
                name=name,
                optimize=optimize,
                head_correction_form=head_correction_form
        )

    def set_harmonic(self, phi0, k):
        """Sets a fixed harmonic dihedral potential.
        Using this method is not compatible force msibi.forces.Force
        objects that are set to be optimized during MSIBI

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
                    "for a force set for optimization. Instead, use either "
                    "set_from_file() or set_quadratic()."
            )
        self.type = "static"
        self.force_init = "harmonic_dihedral = hoomd.md.dihedral.harmonic()"
        self.force_entry = HARMONIC_DIHEDRAL_ENTRY.format(
                self.name, k, d, n, phi0
        )

    def _get_distribution(self, state, gsd_file):
        return dihedral_distribution(
                gsd_file=gsd,
                A_name=self.type1,
                B_name=self.type2,
                C_name=self.type3,
                D_name=self.type4,
                start=-state.n_frames,
                histogram=True,
                normalize=True,
                bins=self.nbins
        )        
