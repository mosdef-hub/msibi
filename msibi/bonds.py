from msibi.utils.sorting import natural_sort

class Bond(object):
    def __init__(self, type1, type2, k, r0):
        self.type1, self.type2 = sorted(
                [type1, type2],
                key=natural_sort
                )
        self.name = f"{self.type1}-{self.type2}"
        self.k = k
        self.r0 = r0
        self._states = dict()

    def _add_state(self, state):
        self._states[state] = {
                "k": self.k,
                "r0": self.r0
            }

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
