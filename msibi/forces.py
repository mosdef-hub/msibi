import math
import os

import numpy as np

from cmeutils.structure import angle_distribution, bond_distribution
from msibi.potentials import quadratic_spring, bond_correction 
from msibi.utils.error_calculation import calc_similarity
from msibi.utils.smoothing import savitzky_golay
from msibi.utils.sorting import natural_sort


HARMONIC_BOND_ENTRY = "harmonic_bond.bond_coeff.set('{}', k={}, r0={})"
FENE_BOND_ENTRY = "fene.bond_coeff.set('{}', k={}, r0={}, sigma={}, epsilon={})"
TABLE_BOND_ENTRY = "btable.set_from_file('{}', '{}')"
HARMONIC_ANGLE_ENTRY = "harmonic_angle.angle_coeff.set('{}', k={}, t0={})"
COSINE_ANGLE_ENTRY = "cosinesq.angle_coeff.set('{}', k={}, t0={})"
TABLE_ANGLE_ENTRY = "atable.set_from_file('{}', '{}')"


class Force(object):
    """Creates a bond potential, either to be held constant, or to be
    optimized.

    Parameters
    ----------
    type1, type2 : str, required
        The name of each particle type in the bond.
        Must match the names found in the State's .gsd trajectory file

    """
    def __init__(self, name, head_correction_form="linear"):
        self.type1, self.type2 = sorted(
                    [type1, type2],
                    key=natural_sort
                )
        self.name = name 
        self._potential_file = "" 
        self.potential = None 
        self.previous_potential = None
        self.head_correction_form = head_correction_form
        self._smoothing_window = 3
        self._smoothing_order = 1
        self._nbins = 100
        self._force_type = None
        self.xmin = None
        self.xmax = None
        self.dx = None
        self.x_range = None
        self.n_points = None
        self._states = dict()

    @property
    def smoothing_window(self):
        return _self.smoothing_window

    @smoothing_window.setter
    def smoothing_window(self, value):
        _self.smoothing_window = value

    @property
    def smoothing_order(self):
        return _self.smoothing_order

    @smoothing_order.setter
    def smoothing_order(self, value):
        _self.smoothing_order = value

    @property
    def nbins(self):
        return self._nbins

    @nbins.setter
    def nbins(self, value):
        self._nbins =  value

    @property
    def target_distribution(self, state):
        return _self.target_distribution(state)
    
    @target_distribution.setter
    def target_distribution(self, state):
        self._target_distribution = array

    @property
    def current_distribution(self, state):
        return self._get_distribution(state)

    @property
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
        self.force_type = "table"
        self.x_min = x_min
        self.x_max = x_max
        self.dx = x_max / n_points
        self.x_range = np.arange(x_min, x_max, self.dx)
        self.potential = quadratic_spring(self.x_range, x0, k4, k3, k2)
        self.n_points = len(self.x_range)
        self.force_init = f"btable = hoomd.md.bond.table(width={self.n_points})"
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
        self.n_points = len(self.l_range)
        self.dx = np.round(self.x_range[1] - self.x_range[0], 3) 
        self.potential = f[:,1]
        self.x_min = self.l_range[0]
        self.x_max = self.l_range[-1] + self.dx

        self.force_type = "table"
        self.force_init = f"btable = hoomd.md.bond.table(width={self.n_points})"
        self.force_entry = TABLE_BOND_ENTRY.format(
                self.name, self._potential_file
        ) 

    def update_potential_file(self, fpath):
        """Set (or reset) the path to a table potential file.
        This function ensures that the Bond.bond_entry attribute
        is correctly updated when a potential file path is generated
        or updated.

        Parameters:
        -----------
        fpath : str, required
            Full path to the text file

        """
        if self.force_type != "table":
            raise RuntimeError("Updating potential file paths can only "
                    "be done for potential types that use table potentials."
            )
        self._potential_file = fpath
        self.force_entry = TABLE_BOND_ENTRY.format(
                self.name, self._potential_file
        )

    def _add_state(self, state):
        """Add a state to be used in optimizing this bond.

        Parameters
        ----------
        state : msibi.state.State
            A State object already created.

        """ #TODO: Set target distribution elsewhere --> Use setter
        self._states[state] = {
                "target_distribution": None,
                "current_distribution": None,
                "alpha": state.alpha,
                "alpha_form": "linear",
                "f_fit": [],
                "path": state.dir
        }

    def _get_state_distribution(self, state, bins, query=False):
        """Find the bond length distribution of a Bond at a State."""
        if query:
            traj = state.query_traj
        else:
            traj = state.traj_file

        return self._get_distribution(gsd_file=traj)

    def _compute_current_distribution(self, state):
        """Find the current bond length distribution of the query trajectory"""
        distribution = self._get_state_distribution(
                state, query=True, bins=self.n_points
        )
        if state._opt.smooth_dist:
            distribution[:,1] = savitzky_golay(
                    distribution[:,1], 3, 1, deriv=0, rate=1
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
        self.previous_potential = np.copy(self.potential)
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
        self.potential = self._correct_potential(
                self.x_range, self.potential, self.head_correction_form
        )


class Bond(Force):
    def __init__(type1, type2, head_correction_form="linear"):
        self.type1, self.type2 = sorted(
                    [type1, type2],
                    key=natural_sort
                )
        name = f"{self.type1}-{self.type2}"
        super(Bond, self).__init__(
                name=name, head_correciton_form=head_correciton_form
        )

    def set_harmonic(self, l0, k):
        pass

    def _get_distribution(self, gsd_file):
        return bond_distribution(
                gsd_file=gsd,
                A_name=self.type1,
                B_name=self.type2,
                start=-state._opt.max_frames,
                histogram=True,
                normalize=True,
                l_min=self.x_min,
                l_max=self.x_max,
                bins=self.nbins
        )        

    def _correct_potential(self):
        pass


class Angle(Force):
    def __init__(type1, type2, type3,  head_correction_form="linear"):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        name = f"{self.type1}-{self.type2}-{self.type3}"
        super(Angle, self).__init__(
                name=name, head_correciton_form=head_correciton_form
        )

    def set_harmonic(self, t0, k):
        pass

    def _get_distribution(self, gsd_file):
        return angle_distribution(
                gsd_file=gsd,
                A_name=self.type1,
                B_name=self.type2,
                C_name=self.type3,
                start=-state._opt.max_frames,
                histogram=True,
                normalize=True,
                l_min=self.x_min,
                l_max=self.x_max,
                bins=self.nbins
        )        



