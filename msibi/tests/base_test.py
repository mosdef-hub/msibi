import mdtraj as md
import numpy as np
import pytest

from msibi.pair import Pair
from msibi.potentials import mie
from msibi.state import State
from msibi.utils.general import get_fn

dr = 0.1 / 6.0
r = np.arange(0, 2.5 + dr, dr)
k_B = 1.9872041e-3  # kcal/mol-K
T = 298.0  # K


class BaseTest:
    @pytest.fixture
    def state0(self):
        return self.init_state(0)

    @pytest.fixture
    def state1(self):
        return self.init_state(1)

    def init_state(self, state_number):
        pair = Pair("0", "1", potential=mie(r, 1.0, 1.0))
        topology_filename = get_fn("final.hoomdxml")
        traj_filename = get_fn(f"state{state_number}/query.dcd")
        t = md.load(traj_filename, top=topology_filename)
        pair_list = t.top.select_pairs('name "0"', 'name "1"')
        rdf_filename = get_fn(f"state{state_number}/target-rdf.txt")
        rdf = np.loadtxt(rdf_filename)
        alpha = 0.5
        state_dir = get_fn(f"state{state_number}/")
        state = State(
            k_B * T, state_dir=state_dir, top_file="sys.hoomdxml", name="state0"
        )
        pair.add_state(state, rdf, alpha, pair_list)
        return pair, state, rdf
