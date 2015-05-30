import pytest

import numpy as np

from msibi.optimize import MSIBI
from msibi.tests.test_pair import init_state


dr = 0.1/6.0
r = np.arange(0, 2.5+dr, dr)
pot_r = np.arange(0, 2.0+dr, dr)
r_range = np.asarray([0.0, 2.5+dr])
n_bins = 151
k_B = 1.9872041e-3  # kcal/mol-K
T = 298.0  # K

def test_msibi_init_single_cutoff():
    opt = MSIBI(2.5, n_bins)
    assert(opt.pot_cutoff == opt.rdf_cutoff)
    assert(opt.n_rdf_points == n_bins)
    assert(opt.rdf_n_bins == n_bins + 1)
    assert(opt.r_switch == 14.6/6.0)
    assert(opt.dr == 0.1/6.0)
    assert(opt.smooth_rdfs == False)
    assert(opt.rdf_r_range.shape[0] == 2)
    assert(opt.pot_r.shape[0] == n_bins)

def test_msibi_init_multiple_cutoff():
    opt = MSIBI(2.5, n_bins, pot_cutoff=2.0)
    assert(opt.pot_cutoff != opt.rdf_cutoff)
    assert(opt.n_rdf_points == n_bins)
    assert(opt.rdf_n_bins == n_bins + 1)
    assert(opt.r_switch == 11.6/6.0)
    assert(opt.dr == 0.1/6.0)
    assert(opt.smooth_rdfs == False)
    assert(opt.rdf_r_range.shape[0] == 2)
    assert(opt.pot_r.shape[0] != n_bins)
    assert(opt.pot_r.shape[0] == 121)

def test_msibi_optimize_states():
    pair, state0, rdf = init_state(0)
    opt = MSIBI(2.5, n_bins, pot_cutoff=2.5)
    opt.optimize([state0], [pair], n_iterations=0, engine='hoomd')
    
