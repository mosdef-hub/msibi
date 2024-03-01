import os
import pytest

import numpy as np

from msibi import Angle, Bond, Pair, State


test_assets = os.path.join(os.path.dirname(__file__), "assets")


class BaseTest:

    @pytest.fixture
    def stateX(self, alpha):
        state = State(
                name="X",
                alpha=alpha,
                kT=1.0,
                traj_file=os.path.join(test_assets, "stateX.gsd")
                n_frames=100
        )
        return state 

    @pytest.fixture
    def stateY(self, alpha):
        state = State(
                name="Y",
                alpha=alpha,
                kT=1.0,
                traj_file=os.path.join(test_assets, "stateY.gsd")
                n_frames=100
        )
        return state 

    @pytest.fixture
    def pairA(self):
        pair = Pair(
                type1="A",
                type2="A",
                r_cut=3.0,
                nbins=100,
                exclude_bonded=True
        )
        pair.set_lj(sigma=2, epsilon=2, r_cut=3.0, r_min=0.1)
        return pair

    @pytest.fixture
    def pairB(self, optimize):
        pair = Pair(
                type1="B",
                type2="B",
                r_cut=3.0,
                nbins=100,
                optimize=optimize,
                exclude_bonded=True
        )
        pair.set_lj(sigma=1.5, epsilon=1, r_cut=3.0, r_min=0.1)
        return pair

    @pytest.fixture
    def pairAB(self, optimize):
        pair = Pair(
                type1="A",
                type2="B",
                r_cut=3.0,
                nbins=100,
                optimize=optimize,
                exclude_bonded=True
        )
        pair.set_lj(sigma=1.5, epsilon=1, r_cut=3.0, r_min=0.1)
        return pair
    
    @pytest.fixture
    def bondAB(self, optimize):
        bond = Bond(
            type1="A",
            type2="B",
            optimize=optimize,
            nbins=100
        )
        bond.set_quadratic(x0=1, k4=0, k3=0, k2=300, x_min=0, x_max=2)
        return bond

    @pytest.fixture
    def angleABA(self, optimize):
        angle = Angle(
                type1="A",
                type2="B",
                type3="A",
                optimize=optimize,
                nbins=100
        )
        angle.set_quadratic(x0=1, k4=0, k3=0, k2=200, x_min=0, x_max=np.pi)
        return angle

    @pytest.fixture
    def rdfAA(self):
        return self.get_rdf(0)

    @pytest.fixture
    def rdfBB(self):
        return self.get_rdf(0)

    @pytest.fixture
    def rdfAB(self):
        return self.get_rdf(0)
