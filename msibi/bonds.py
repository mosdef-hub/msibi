import math
import os

import numpy as np

from cmeutils.structure import ( 
        angle_distribution, bond_distribution, dihedral_distribution
)
from msibi.potentials import quadratic_spring, bond_correction 
from msibi.utils.error_calculation import calc_similarity
from msibi.utils.smoothing import savitzky_golay
from msibi.utils.sorting import natural_sort


HARMONIC_BOND_ENTRY = "harmonic_bond.bond_coeff.set('{}', k={}, r0={})"
FENE_BOND_ENTRY = "fene.bond_coeff.set('{}', k={}, r0={}, sigma={}, epsilon={})"
TABLE_BOND_ENTRY = "btable.set_from_file('{}', '{}')"
HARMONIC_ANGLE_ENTRY = "harmonic_angle.angle_coeff.set('{}', k={}, t0={})"
COSINE_ANGLE_ENTRY = "cosinesq.angle_coeff.set('{}', k={}, t0={})"
HARMONIC_DIHEDRAL_ENTRY = "harmonic_dihedral.dihedral_coeff.set('{}', k={}, d={}, n={}, phi0={})"
TABLE_ANGLE_ENTRY = "atable.set_from_file('{}', '{}')"
TABLE_DIHEDRAL_ENTRY = "dtable.set_from_file('{}', '{}')"


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
        self.l_range = f[:,0]
        self.n_points = len(self.l_range)
        self.dl = np.round(self.l_range[1] - self.l_range[0], 3) 
        self.potential = f[:,1]
        self.l_min = self.l_range[0]
        self.l_max = self.l_range[-1] + self.dl

        self.bond_type = "table"
        self.bond_init = f"btable = hoomd.md.bond.table(width={self.n_points})"
        self.bond_entry = TABLE_BOND_ENTRY.format(
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
        if self.bond_type != "table":
            raise RuntimeError("Updating potential file paths can only "
                    "be done for bond potential types that use table potentials."
            )
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
            if state._opt.smooth_dist:
                target_distribution[:,1] = savitzky_golay(
                        target_distribution[:,1], 3, 1, deriv=0, rate=1
                )
                negative_idx = np.where(target_distribution[:,1] < 0)[0]
                target_distribution[:,1][negative_idx] = 0

            fname = f"bond_dist_{self.name}-state_{state.name}-target.txt"
            fpath = os.path.join(state.dir, fname)
            np.savetxt(fpath, target_distribution)
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
                normalize=True,
                l_min=self.l_min,
                l_max=self.l_max,
                bins=bins
        )

    def _compute_current_distribution(self, state):
        """Find the current bond length distribution of the query trajectory"""
        bond_distribution = self._get_state_distribution(
                state, query=True, bins=self.n_points
        )
        if state._opt.smooth_dist:
            bond_distribution[:,1] = savitzky_golay(
                    bond_distribution[:,1], 3, 1, deriv=0, rate=1
            )
            negative_idx = np.where(bond_distribution[:,1] < 0)[0]
            bond_distribution[:,1][negative_idx] = 0
        self._states[state]["current_distribution"] = bond_distribution

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
        fname = f"bond_dist_{self.name}-state_{state.name}-step_{iteration}.txt"
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
    def __init__(self, type1, type2, type3, head_correction_form="linear"):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.name = f"{self.type1}-{self.type2}-{self.type3}"
        self.head_correction_form = head_correction_form
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
        self.theta_min = 0
        self.theta_max = math.pi
        self.potential = quadratic_spring(self.theta_range, theta0, k4, k3, k2)
        self.n_points = len(self.theta_range)
        self.angle_init = f"atable = hoomd.md.angle.table(width={self.n_points})"
        self.angle_entry = TABLE_ANGLE_ENTRY.format(
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
        self.theta_range = f[:,0]
        self.dtheta = np.round(self.theta_range[1] - self.theta_range[0], 3) 
        self.n_points = len(self.theta_range)
        self.potential = f[:,1]
        self.theta_min = 0
        self.theta_max = math.pi

        self.angle_type = "table"
        self.angle_init = f"atable = hoomd.md.angle.table(width={self.n_points})"
        self.angle_entry = TABLE_ANGLE_ENTRY.format(
                self.name, self._potential_file
        ) 

    def update_potential_file(self, fpath):
        """Set (or reset) the path to a table potential file.
        This function ensures that the Angle.angle_entry attribute
        is correctly updated when a potential file path is generated
        or updated.

        Parameters:
        -----------
        fpath : str, required
            Full path to the text file

        """
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
            if state._opt.smooth_dist:
                target_distribution[:,1] = savitzky_golay(
                        target_distribution[:,1], 3, 1, deriv=0, rate=1
                )
                negative_idx = np.where(target_distribution[:,1] < 0)[0]
                target_distribution[:,1][negative_idx] = 0
                
            fname = f"angle_dist_{self.name}-state_{state.name}-target.txt"
            fpath = os.path.join(state.dir, fname)
            np.savetxt(fpath, target_distribution)
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
                normalize=True,
                theta_min=self.theta_min,
                theta_max=self.theta_max,
                bins=bins
        )

    def _compute_current_distribution(self, state):
        """Find the current bond angle distribution of the query trajectory"""
        angle_distribution = self._get_state_distribution(
                state, query=True, bins=self.n_points
        )
        if state._opt.smooth_dist:
            angle_distribution[:,1] = savitzky_golay(
                    angle_distribution[:,1], 3, 1, deriv=0, rate=1
            )
            negative_idx = np.where(angle_distribution[:,1] < 0)[0]
            angle_distribution[:,1][negative_idx] = 0
        self._states[state]["current_distribution"] = angle_distribution

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
        
        fname = f"angle_dist_{self.name}-state_{state.name}-step_{iteration}.txt"
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
                    kT * np.log(current_dist[:,1] / target_dist[:,1]) / N
            )
        # Apply corrections
        self.potential = bond_correction(
                self.theta_range,
                self.potential,
                self.head_correction_form
        )


class Dihedral(object):
    """Creates a bond dihedral potential, either to be held constant, or to be
    optimized.

    Parameters
    ----------
    type1, type2, type3, type4 : str, required
        The name of each particle type in the dihedral.
        Must match the names found in the State's .gsd trajectory file

    """
    def __init__(self, type1, type2, type3, type4, head_correction_form="linear"):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.type4 = type4
        self.name = f"{self.type1}-{self.type2}-{self.type3}-{self.type4}"
        self.head_correction_form = head_correction_form
        self._potential_file = ""
        self.potential = None
        self.previous_potential = None
        self._states = dict()

    def set_harmonic(self, k, d, n, phi0):
        """Creates a hoomd.md.dihedral.harmonic() type of bond dihedral 
        potential to be used during the query simulations.
        This method is not compatible when optimizing bond dihedral potentials.
        Rather, it should be used to set a static potential while
        optimizing Pairs, Bonds, or Angles.

        Parameters
        ----------
        k : float, required
            The potential constant
        d : int, required
            Sign factor
        n : int, required
            Angle scaling factor
        phi0 : float, required
            The equilibrium resting angle

        """
        self.dihedral_type = "static"
        self.dihedral_init = "harmonic_dihedral = hoomd.md.dihedral.harmonic()"
        self.angle_entry = HARMONIC_DIHEDRAL_ENTRY.format(self.name, k, d, n, phi0) 

    def set_quadratic(self, phi0, k4, k3, k2, n_points=100):
        """Set a bond dihedral potential based on the following function:

            V(theta) = k4(phi-phi0)^4 + k3(phi-phi0)^3 + k2(phi-phi0)^2

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
        self.dihedral_type = "table"
        self.dphi = 2*math.pi / (n_points - 1)
        self.phi_range = np.arange(-math.pi, math.pi + self.dphi, self.dphi)
        self.phi_min = -math.pi 
        self.phi_max = math.pi
        self.potential = quadratic_spring(self.phi_range, phi0, k4, k3, k2)
        self.n_points = len(self.phi_range)
        self.dihedral_init = f"dtable = hoomd.md.dihedral.table(width={self.n_points})"
        self.dihedral_entry = TABLE_DIHEDRAL_ENTRY.format(
                self.name, self._potential_file
        ) 

    def set_from_file(self, file_path):
        """Creates a bond dihedral potential from a text file.
        The columns of the text file must be in the order of phi, V, F
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
        self.phi_range = f[:,0]
        self.dphi = np.round(self.phi_range[1] - self.phi_range[0], 3) 
        self.n_points = len(self.phi_range)
        self.potential = f[:,1]
        self.phi_min = -math.pi 
        self.phi_max = math.pi

        self.dihedral_type = "table"
        self.dihedral_init = f"dtable = hoomd.md.dihedral.table(width={self.n_points})"
        self.dihedral_entry = TABLE_DIHEDRAL_ENTRY.format(
                self.name, self._potential_file
        ) 

    def update_potential_file(self, fpath):
        """Set (or reset) the path to a table potential file.
        This function ensures that the Angle.angle_entry attribute
        is correctly updated when a potential file path is generated
        or updated.

        Parameters:
        -----------
        fpath : str, required
            Full path to the text file

        """
        self._potential_file = fpath
        self.dihedral_entry = TABLE_DIHEDRAL_ENTRY.format(
                self.name, self._potential_file
        )

    def _add_state(self, state):
        """Add a state to be used in optimizing this angle.

        Parameters
        ----------
        state : msibi.state.State
            A State object already created

        """
        if state._opt.optimization == "dihedrals":
            target_distribution = self._get_state_distribution(
                    state, query=False, bins=self.n_points
            )
            if state._opt.smooth_dist:
                target_distribution[:,1] = savitzky_golay(
                        target_distribution[:,1], 5, 1, deriv=0, rate=1
                )
                negative_idx = np.where(target_distribution[:,1] < 0)[0]
                target_distribution[:,1][negative_idx] = 0
                
            fname = f"dihedral_dist_{self.name}-state_{state.name}-target.txt"
            fpath = os.path.join(state.dir, fname)
            np.savetxt(fpath, target_distribution)
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
        """Finds the distribution of dihedrals for a given type"""
        if query:
            traj = state.query_traj
        else:
            traj = state.traj_file
        return dihedral_distribution(
                gsd_file=traj,
                A_name=self.type1,
                B_name=self.type2,
                C_name=self.type3,
                D_name=self.type4,
                start=-state._opt.max_frames,
                histogram=True,
                normalize=True,
                bins=bins
        )

    def _compute_current_distribution(self, state):
        """Find the current bond angle distribution of the query trajectory"""
        dihedral_distribution = self._get_state_distribution(
                state, query=True, bins=self.n_points
        )
        if state._opt.smooth_dist:
            dihedral_distribution[:,1] = savitzky_golay(
                    dihedral_distribution[:,1], 5, 1, deriv=0, rate=1
            )
            negative_idx = np.where(dihedral_distribution[:,1] < 0)[0]
            dihedral_distribution[:,1][negative_idx] = 0
        self._states[state]["current_distribution"] = dihedral_distribution

        f_fit = calc_similarity(
                dihedral_distribution[:,1],
                self._states[state]["target_distribution"][:,1] 
        )
        self._states[state]["f_fit"].append(f_fit)

    def _save_current_distribution(self, state, iteration):
        """Save the current bond dihedral distribution 

        Parameters
        ----------
        state : State
            A state object
        iteration : int
            Current iteration step, used in the filename

        """
        distribution = self._states[state]["current_distribution"]
        distribution[:,0] -= self.dphi / 2
        
        fname = f"dihedral_dist_{self.name}-state_{state.name}-step_{iteration}.txt"
        fpath = os.path.join(state.dir, fname)
        np.savetxt(fpath, distribution)

    def _update_potential(self):
        """Compare distributions of current iteration against target,
        and update the Dihedral potential via Boltzmann inversion.

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
                self.phi_range,
                self.potential,
                self.head_correction_form
        )

