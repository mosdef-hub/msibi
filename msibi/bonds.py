from cmeutils.structure import angle_distribution, bond_distribution
from msibi.utils.sorting import natural_sort
from msibi.utils.error_calculation import calc_similarity


HARMONIC_BOND_ENTRY = "haromonic_bond.bond_coeff.set('{}', k={}, r0={})"
TABLE_BOND_ENTRY = "btable.bond_coeff.set('{}', {})"
HARMONIC_ANGLE_ENTRY = "harmonic_angle.angle_coeff.set('{}', k={}, t0={})"
TABLE_ANGLE_ENTRY = "atable.angle_coeff.set('{}', {})"


class Bond(object):
    """Creates a bond potential, either to be held constant, or to be
    optimized.

    Parameters
    ----------
    type1, type2 : str, required
        The name of each particle type in the bond.
        Must match the names found in the State's .gsd trajectory file

    """
    def __init__(self, type1, type2):
        self.type1, self.type2 = sorted(
                    [type1, type2],
                    key=natural_sort
                )
        self.name = f"{self.type1}-{self.type2}"
        self.potential_file = "" 
        self.potential = None 
        self.previous_potential = None
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
        self.bond_type = "harmonic"
        self.bond_init = "harmonic_bond = hoomd.md.bond.harmonic()"
        self.bond_entry = HARMONIC_BOND_ENTRY.format(self.name, k, l0)
    
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
        def create_bond_table(l, l0, k4, k3, k2):
            L = l - l0
            V_l = (k4*(L))**4 + (k3*(L))**3 + (k2*(L))**2
            return V_l

        self.bond_type = "quadratic"
        self.dl = (l_max - l_min) / n_points
        self.l_range = np.arange(l_min, l_max + self.dl, self.dl)
        self.potential = create_bond_table(self.l_range, l0, k4, k3, k2)
        self.bond_init = f"btable = bond.table(width={n_points})"
        self.bond_entry = TABLE_BOND_ENTRY.format(
                self.name, self.potential_file
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
                    state, query=False
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

    def _get_state_distribution(self, state, query=False):
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
                histogram=True
        )

    def _compute_current_distribution(self, state):
        """Find the current bond length distribution of the query trajectory"""
        bond_distribution = self.get_state_distribution(state, query=True)
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
        """
        Compare distributions of current iteration against target,
        and update the Bond potential via Boltzmann inversion.

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
        self.potential_file = ""
        self.potential = None
        self.previous_potential = None
        self._states = dict()

    def set_harmonic(self, k, theta0):
        self.angle_type = "harmonic"
        self.angle_init = "harmonic_angle = hoomd.md.angle.harmonic()"
        self.angle_entry = HARMONIC_ANGLE_ENTRY.format(self.name, k, theta0) 

    def set_quadratic(
            self, theta0, k4, k3, k2, theta_min, theta_max, n_points=101
    ):
        """Set a bond potential based on the following function:

            V(theta) = k4(theta-theta0)^4 + k3(theta-theta0)^3 + k2(theta-theta0)^2

        Using this method will create a table potential V(theta) over the range
        theta_min - theta_max.

        This should be the angle potential form of choice when optimizing angles 
        as opposed to using `set_harmonic`. However, you can also use this
        method to set a static angle potential while you are optimizing other
        potentials such as Bonds or Pairs.

        Parameters
        ----------
        theta0, k4, k3, k2 : float, required
            The paraters used in the V(theta) function described above
        theta_min : float, required
            The lower bound of the angle potential angles 
        theta_max : float, required
            The upper bound of the angle potential angles
        n_points : int, default = 101 
            The number of points between theta_min-theta_max used to create
            the table potential

        """
        def create_angle_table(theta, theta0, k4, k3, k2):
            T = theta - theta0
            V_theta = (k4*(T))**4 + (k3*(T))**3 + (k2*(T))**2
            return V_theta

        self.angle_type = "quadratic"
        self.dtheta = (theta_max - theta_min) / n_points
        self.theta_range = np.arange(
                theta_min, theta_max + self.dtheta, self.dtheta
        )
        self.potential = create_angle_table(
                self.theta_range, theta0, k4, k3, k2
        )
        self.angle_init = f"atable = angle.table(width={n_points})"
        self.angle_entry = TABLE_ANGLE_ENTRY.format(
                self.name, self.potential_file
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
                    state, query=False
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

    def _get_state_distribution(self, state, query=False):
        """Finds the distribution of angles for a given Angle"""
        if query:
            traj = state.query_traj
        else:
            traj = state.traj_file
        return angle_distribution(
                gsd_file=traj,
                A_name=self.type1,
                B_name=self.type2,
                start=-state._opt.max_frames,
                histogram=True
        )

    def _compute_current_distribution(self, state):
        """Find the current bond angle distribution of the query trajectory"""
        angle_distribution = self._get_state_distribution(state, query=True)
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
        """
        Compare distributions of current iteration against target,
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

