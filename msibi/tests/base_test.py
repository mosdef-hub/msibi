import os
import pytest

import numpy as np

from msibi import Angle, Bond, Dihedral, Pair, State


test_assets = os.path.join(os.path.dirname(__file__), "assets")


class BaseTest:

    @pytest.fixture
    def stateX(self, tmp_path):
        state = State(
                name="X",
                alpha=1.0,
                kT=1.0,
                traj_file=os.path.join(test_assets, "stateX.gsd"),
                n_frames=100,
                _dir=tmp_path
        )
        return state 

    @pytest.fixture
    def stateY(self, tmp_path):
        state = State(
                name="Y",
                alpha=1.0,
                kT=1.0,
                traj_file=os.path.join(test_assets, "stateY.gsd"),
                n_frames=100,
                _dir=tmp_path
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
    def pairB(self):
        pair = Pair(
                type1="B",
                type2="B",
                r_cut=3.0,
                nbins=100,
                optimize=False,
                exclude_bonded=True
        )
        pair.set_lj(sigma=1.5, epsilon=1, r_cut=3.0, r_min=0.1)
        return pair

    @pytest.fixture
    def pairAB(self):
        pair = Pair(
                type1="A",
                type2="B",
                r_cut=3.0,
                nbins=100,
                optimize=False,
                exclude_bonded=True
        )
        pair.set_lj(sigma=1.5, epsilon=1, r_cut=3.0, r_min=0.1)
        return pair
    
    @pytest.fixture
    def bond(self):
        bond = Bond(
            type1="A",
            type2="B",
            optimize=False,
            nbins=100
        )
        return bond

    @pytest.fixture
    def angle(self):
        angle = Angle(
                type1="A",
                type2="B",
                type3="A",
                optimize=False,
                nbins=100
        )
        return angle

    @pytest.fixture
    def dihedral(self):
        dihedral = Dihedral(
                    type1="A",
                    type2="B",
                    type3="A",
                    type4="B",
                    optimize=False,
                    nbins=100
        )
        return dihedral 

    @pytest.fixture
    def rdfAA(self):
        return self.get_rdf(0)

    @pytest.fixture
    def rdfBB(self):
        pass

    @pytest.fixture
    def rdfAB(self):
        pass
