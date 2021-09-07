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
    def state0(self, tmp_path):
        return self.init_state(0, tmp_path)

    @pytest.fixture
    def state1(self, tmp_path):
        return self.init_state(1, tmp_path)

    @pytest.fixture
    def pair(self):
        return Pair("0", "1", potential=mie(r, 1.0, 1.0))

    def init_state(self, state_n, tmp_path):
        traj_filename = os.path.join(test_assets, f"query{state_n}.gsd")
        alpha = 0.5
        state = State(
            name="state0",
            kT=k_B * T,
            traj_file=traj_filename,
            _dir=tmp_path
        )
        return state
