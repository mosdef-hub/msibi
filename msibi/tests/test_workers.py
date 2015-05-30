import pytest

import numpy as np

from msibi.optimize import MSIBI
from msibi.tests.test_pair import init_state
from msibi.utils.exceptions import UnsupportedEngine
from msibi.workers import run_query_simulations


dr = 0.1/6.0
r = np.arange(0, 2.5+dr, dr)
pot_r = np.arange(0, 2.0+dr, dr)
r_range = np.asarray([0.0, 2.5+dr])
n_bins = 151
k_B = 1.9872041e-3  # kcal/mol-K
T = 298.0  # K

def test_unsupported_engine():
    pair, state0, rdf = init_state(0)
    _, state1, rdf = init_state(1)
    engine = 'crammps'
    with pytest.raises(UnsupportedEngine):
        run_query_simulations([state0, state1], engine=engine)
