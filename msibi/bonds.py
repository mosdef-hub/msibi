import math
import os

import numpy as np

from cmeutils.structure import angle_distribution, bond_distribution
from msibi.utils.sorting import natural_sort
from msibi.potentials import quadratic_spring, bond_correction 
from msibi.utils.error_calculation import calc_similarity


HARMONIC_BOND_ENTRY = "harmonic_bond.bond_coeff.set('{}', k={}, r0={})"
FENE_BOND_ENTRY = "fene.bond_coeff.set('{}', k={}, r0={}, sigma={}, epsilon={})"
TABLE_BOND_ENTRY = "btable.set_from_file('{}', '{}')"
HARMONIC_ANGLE_ENTRY = "harmonic_angle.angle_coeff.set('{}', k={}, t0={})"
COSINE_ANGLE_ENTRY = "cosinesq.angle_coeff.set('{}', k={}, t0={})"
TABLE_ANGLE_ENTRY = "atable.set_from_file('{}', '{}')"


class Bond(object):
    """Creates a bond potential, either to be held constant, or to be
    optimized.

    Parameters
    ----------
    type1, type2 : str, required
        The name of each particle type in the bond.
        Must match the names found in the State's .gsd trajectory file

    """
    def __init__(self, type1, type2, head_correction_form="linear"):
        self.type1, self.type2 = sorted(
                    [type1, type2],
                    key=natural_sort
                )
        self.name = f"{self.type1}-{self.type2}"
        self._potential_file = "" 
        self.potential = None 
        self.previous_potential = None
        self.head_correction_form = head_correction_form
        self._states = dict()
    
    def set_harmonic(self, k, l0):
        """Creates a hoomd.md.bond.harmonic type of bond potential
        to be used during the query simulations. This method is
        not compatible when optimizing bond potentials. Rather,
        this method should only be used to create static bond potentials
        while optimizing Pairs or Angles.

        See the `set_quadratic` method for another option.

        Parameters
        ----------
        l0 : float, required
            The equilibrium bond length
        k : float, required
            The spring constant

        """
        self.bond_type = "static"
        self.bond_init = "harmonic_bond = hoomd.md.bond.harmonic()"
        self.bond_entry = HARMONIC_BOND_ENTRY.format(self.name, k, l0)

    def set_fene(self, k, r0, epsilon, sigma):
        """Creates a hoomd.md.bond.fene type of bond potential
        to be used during the query simulations. This method is
        not compatible when optimizing bond stretching potentials.
        Rather, this method should only be used to create static bond
        stretching potentials while optimizing Pairs or Angles.

        """
        self.bond_type = "static"
        self.bond_init = "fene = bond.fene()"
        self.bond_entry = FENE_BOND_ENTRY.format(
                self.name, k, r0, sigma, epsilon
        )
    
    def set_quadratic(self, l0, k4, k3, k2, l_min, l_max, n_points=101):
        """Set a bond potential based on the following function:

            V(l) = k4(l-l0)^4 + k3(l-l0)^3 + k2(l-l0)^2

        Using this method will create a table potential V(l) over the range
        l_min - l_max.

        This should be the bond potential form of choice when optimizing bonds
        as opposed to using `set_harmonic`. However, you can also use this
        method to set a static bond potential while you are optimizing other
        potentials such as Angles or Pairs.

        Parameters
        ----------
        l0, k4, k3, k2 : float, required
            The paraters used in the V(l) function described above
        l_min : float, required
            The lower bound of the bond potential lengths
        l_max : float, required
            The upper bound of the bond potential lengths
        n_points : int, default = 101 
            The number of points between l_min-l_max used to create
            the table potential

        """
        self.bond_type = "table"
        self.l_min = l_min
        self.l_max = l_max
        self.dl = l_max / n_points
        self.l_range = np.arange(l_min, l_max, self.dl)
        self.potential = quadratic_spring(self.l_range, l0, k4, k3, k2)
        self.n_points = len(self.l_range)
        self.bond_init = f"btable = hoomd.md.bond.table(width={self.n_points})"
        self.bond_entry = TABLE_BOND_ENTRY.format(
                self.name, self._potential_file
        ) 

    def update_potential_file(self, fpath):
        #TODO Throw error if self.bond_type isn't one that uses files (table)
        self._potential_file = fpath
        self.bond_entry = TABLE_BOND_ENTRY.format(
                self.name, self._potential_file
        )

    def _add_state(self, state):
        """Add a state to be used in optimizing this bond.

        Parameters
        ----------
        state : msibi.state.State
            A State object already created.

        """
        if state._opt.optimization == "bonds":
            target_distribution = self._get_state_distribution(
                    state, query=False, bins=self.n_points
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

    def _get_state_distribution(self, state, bins, query=False):
        """Find the bond length distribution of a Bond at a State."""
        if query:
            traj = state.query_traj
        else:
            traj = state.traj_file

        return bond_distribution(
                gsd_file=traj,
                A_name=self.type1,
                B_name=self.type2,
                start=-state._opt.max_frames,
                histogram=True,
                l_min=self.l_min,
                l_max=self.l_max,
                bins=bins
        )

    def _compute_current_distribution(self, state):
        """Find the current bond length distribution of the query trajectory"""
        bond_distribution = self._get_state_distribution(
                state, query=True, bins=self.n_points
        )
        self._states[state]["current_distribution"] = bond_distribution
        # TODO ADD SMOOTHING
        f_fit = calc_similarity(
                    bond_distribution[:,1],
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
        distribution[:,0] -= self.dl / 2
        fname = f"bond_pot_{self.name}-state_{state.name}-step_{iteration}.txt"
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

        # Apply corrections
        self.potential = bond_correction(
                self.l_range,
                self.potential,
                self.head_correction_form
        )


class Angle(object):
    """Creates a bond angle potential, either to be held constant, or to be
    optimized.

    Parameters
    ----------
    type1, type2, type3 : str, required
        The name of each particle type in the bond.
        Must match the names found in the State's .gsd trajectory file

    """
    def __init__(self, type1, type2, type3):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.name = f"{self.type1}-{self.type2}-{self.type3}"
        self._potential_file = ""
        self.potential = None
        self.previous_potential = None
        self._states = dict()

    def set_harmonic(self, k, theta0):
        """Creates a hoomd.md.angle.harmonic() type of bond angle
        potential to be used during the query simulations.
        This method is not compatible when optimizing bond angle potentials.
        Rather, it should be used to set a static angle potential while
        optimizing Pairs or Bonds.

        Parameters
        ----------
        k : float, required
            The potential constant
        theta0 : float, required
            The equilibrium resting angle

        """
        self.angle_type = "static"
        self.angle_init = "harmonic_angle = hoomd.md.angle.harmonic()"
        self.angle_entry = HARMONIC_ANGLE_ENTRY.format(self.name, k, theta0) 

    def set_cosinesq(self, k, theta0):
        """Creates a hoomd.md.angle.cosinesq() type of bond angle
        potential to be used during the query simulations.
        This method is not compatible when optimizing bond angle potentials.
        Rather, it should be used to set a static angle potential while
        optimizing Pairs or Bonds.
        
        Parameters
        ----------
        k : float, required
            The potential constant
        theta0 : float, required
            The equilibrium resting angle

        """
        self.angle_type = "static"
        self.angle_init = "cosinesq = angle.cosinesq()"
        self.angle_entry = COSINE_ANGLE_ENTRY.format(self.name, k, theta0)

    def set_quadratic(self, theta0, k4, k3, k2, n_points=100):
        """Set a bond angle potential based on the following function:

            V(theta) = k4(theta-theta0)^4 + k3(theta-theta0)^3 + k2(theta-theta0)^2

        Using this method will create a table potential V(theta) over the range
        theta_min - theta_max.

        The angle table potential will range from theta = 0 to theta = math.pi

        This should be the angle potential form of choice when optimizing angles 
        as opposed to using `set_harmonic`. However, you can also use this
        method to set a static angle potential while you are optimizing other
        potentials such as Bonds or Pairs.

        Parameters
        ----------
        theta0, k4, k3, k2 : float, required
            The paraters used in the V(theta) function described above
        n_points : int, default = 101 
            The number of points between theta_min-theta_max used to create
            the table potential

        """
        self.angle_type = "table"
        self.dtheta = math.pi / (n_points - 1)
        self.theta_range = np.arange(0, math.pi + self.dtheta, self.dtheta)
        self.potential = quadratic_spring(self.theta_range, theta0, k4, k3, k2)
        self.n_points = len(self.theta_range)
        self.angle_init = f"atable = hoomd.md.angle.table(width={self.n_points})"
        self.angle_entry = TABLE_ANGLE_ENTRY.format(
                self.name, self._potential_file
        ) 

    def update_potential_file(self, fpath):
        #TODO Throw error if self.angle_type isn't one that uses fils (table)
        self._potential_file = fpath
        self.angle_entry = TABLE_ANGLE_ENTRY.format(
                self.name, self._potential_file
        )

    def _add_state(self, state):
        """Add a state to be used in optimizing this angle.

        Parameters
        ----------
        state : msibi.state.State
            A State object already created

        """
        if state._opt.optimization == "angles":
            target_distribution = self._get_state_distribution(
                    state, query=False, bins=self.n_points
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

    def _get_state_distribution(self, state, bins, query=False):
        """Finds the distribution of angles for a given Angle"""
        if query:
            traj = state.query_traj
        else:
            traj = state.traj_file
        return angle_distribution(
                gsd_file=traj,
                A_name=self.type1,
                B_name=self.type2,
                C_name=self.type3,
                start=-state._opt.max_frames,
                histogram=True,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                bins=bins
        )

    def _compute_current_distribution(self, state):
        """Find the current bond angle distribution of the query trajectory"""
        angle_distribution = self._get_state_distribution(
                state, query=True, bins=self.n_points
        )
        self._states[state]["current_distribution"] = angle_distribution
        # TODO ADD SMOOTHING
        f_fit = calc_similarity(
                angle_distribution[:,1],
                self._states[state]["target_distribution"][:,1] 
        )
        self._states[state]["f_fit"].append(f_fit)

    def _save_current_distribution(self, state, iteration):
        """Save the current bond angle distribution 

        Parameters
        ----------
        state : State
            A state object
        iteration : int
            Current iteration step, used in the filename

        """
        distribution = self._states[state]["current_distribution"]
        distribution[:,0] -= self.dtheta / 2
        
        fname = f"angle_pot_{self.name}-state_{state.name}-step_{iteration}.txt"
        fpath = os.path.join(state.dir, fname)
        np.savetxt(fpath, distribution)

    def _update_potential(self):
        """Compare distributions of current iteration against target,
        and update the Angle potential via Boltzmann inversion.

        """
        self.previous_potential = np.copy(self.potential)
        for state in self._states:
            kT = state.kT
            current_dist = self._states[state]["current_distribution"]
            target_dist = self._states[state]["target_distribution"]
            N = len(self._states)
            self.potential += state.alpha * (
                    kT * np.log(current_dist[:,1] / target_dist[:,1] / N)
            )

