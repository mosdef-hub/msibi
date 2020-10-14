import pytest

from msibi.optimize import MSIBI
from msibi.tests.test_pair import init_state


n_bins = 151


def test_msibi_init_single_cutoff():
    opt = MSIBI(2.5, n_bins)
    assert opt.pot_cutoff == opt.rdf_cutoff
    assert opt.n_rdf_points == n_bins
    assert opt.rdf_n_bins == n_bins
    assert opt.r_switch == 14.6 / 6.0
    assert opt.dr == 0.1 / 6.0
    assert opt.smooth_rdfs is False
    assert opt.rdf_r_range.shape[0] == 2
    assert opt.pot_r.shape[0] == n_bins


def test_msibi_init_multiple_cutoff():
    opt = MSIBI(2.5, n_bins, pot_cutoff=2.0)
    assert opt.pot_cutoff != opt.rdf_cutoff
    assert opt.n_rdf_points == n_bins
    assert opt.rdf_n_bins == n_bins
    assert opt.r_switch == 11.6 / 6.0
    assert opt.dr == 0.1 / 6.0
    assert opt.smooth_rdfs is False
    assert opt.rdf_r_range.shape[0] == 2
    assert opt.pot_r.shape[0] != n_bins
    assert opt.pot_r.shape[0] == 121


def test_msibi_optimize_states():
    pair, state0, rdf = init_state(0)
    opt = MSIBI(2.5, n_bins, pot_cutoff=2.5)
    opt.optimize([state0], [pair], n_iterations=0, engine="hoomd")


def test_rdf_length():
    pair, state0, rdf = init_state(0)
    opt = MSIBI(2.5, n_bins + 1, pot_cutoff=2.5)
    with pytest.raises(ValueError):
        opt.optimize([state0], [pair], n_iterations=0, engine="hoomd")
