import os
import pytest

import numpy as np

from msibi.pair import Pair
from msibi.potentials import mie
from msibi.state import State


dr = 0.1 / 6.0
r = np.arange(0, 2.5 + dr, dr)
k_B = 1.9872041e-3  # kcal/mol-K
T = 298.0  # K
test_assets = os.path.join(os.path.dirname(__file__), "assets")


class BaseTest:
    @pytest.fixture
    def state0(self):
        return self.init_state(0)

    @pytest.fixture
    def state1(self):
        return self.init_state(1)

    def init_state(self, state_n):
        pair = Pair("0", "1", potential=mie(r, 1.0, 1.0))
        traj_filename = os.path.join(test_assets, f"query{state_n}.gsd")
        rdf_filename = os.path.join(test_assets, f"target-rdf{state_n}.txt")
        rdf = np.loadtxt(rdf_filename)
        alpha = 0.5
        state_dir = get_fn(f"state{state_number}/")
        state = State(
            k_B * T,
            state_dir=state_dir,
            traj_file=traj_filename,
            top_file="sys.hoomdxml",
            name="state0"
        )
        pair.add_state(state, rdf, alpha, pair_list)
        return pair, state, rdf
