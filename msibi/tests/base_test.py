import os
import pytest

import numpy as np

from msibi import Pair, mie, State


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

    @pytest.fixture
    def rdf0(self):
        return self.get_rdf(0)

    @pytest.fixture
    def rdf1(self):
        return self.get_rdf(1)

    def get_rdf(self, state_n):
        return np.loadtxt(os.path.join(test_assets, f"target-rdf{state_n}.txt"))

    def init_state(self, state_n, tmp_path):
        traj_filename = os.path.join(test_assets, f"query{state_n}.gsd")
        state = State(
            name=f"state{state_n}",
            kT=k_B * T,
            alpha=0.5,
            traj_file=traj_filename,
            _dir=tmp_path
        )
        return state
