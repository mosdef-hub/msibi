from cmeutils.structure import angle_distribution, bond_distribution
from msibi.utils.sorting import natural_sort


class Bond(object):
    def __init__(self, type1, type2):
        self.type1, self.type2 = sorted(
                [type1, type2],
                key=natural_sort
                )
        self.name = f"{self.type1}-{self.type2}"
        self._states = dict()
    
    def set_harmonic(self, k, r0):
        """
        """
        self.k = k
        self.r0 = r0
        self.bond_parms = {"k":self.k, "r0":self.r0}
        self.bond_type = "harmonic"
        self.script = ""

    def set_fene(self, k, r0, epsilon, sigma):
        """
        """
        self.k = k
        self.r0 = r0
        self.epsilon = epsilon
        self.sigma = sigma
        self.bond_parms = {
                "k": self.k,
                "r0": r0,
                "epsilon": self.epsilon,
                "sigma": self.sigma
            }
        self.bond_type = "fene"
        self.script = ""

    def set_table(self, file=None, func=None):
        """
        """
        if [file, func].count(None) == 0:
            raise ValueError("Choose one of `file` of `func` to create
                        a table potential")
        self.file = file
        self.func = func
        self.bond_type = "table"
        self.script = ""

    def get_distribution(self, state, query=False):
        if query:
            traj = state.query_traj
        else:
            traj = state.traj_file
        bonds = bond_distribution(traj, self.type1, self.type2)  

    def _add_state(self, state):
        target_distribution = self.get_distribution(state, query=False) 
        self._states[state] = {
                "target_distribution": target_distribution,
                "current_distribution": None,
                "alpha": state.alpha,
                "alpha_form": "linear",
                "f_fit": [],
                "path": state.dir
            }
        self._states[state].update(self.bond_params)


class Angle(object):
    def __init__(self, type1, type2, type3, k, theta):
        self.type1 = type1
        self.type2 = type2
        self.type3 = type3
        self.name = f"{self.type1}-{self.type2}-{self.type3}"
        self.k = k
        self.theta = theta
        self._states = dict()

    def _add_state(self, state):
        self._states[state] = {
                "k": self.k,
                "theta": self.theta
            }
