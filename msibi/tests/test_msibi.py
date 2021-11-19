import pytest

from msibi import MSIBI

from .base_test import BaseTest

n_bins = 151


class TestMSIBI(BaseTest):
    def test_msibi_init_single_cutoff(self):
        opt = MSIBI(2.5, n_bins)
        assert opt.pot_cutoff == opt.rdf_cutoff
        assert opt.n_rdf_points == n_bins
        assert opt.rdf_n_bins == n_bins
        assert opt.r_switch == opt.pot_r[-5]
        assert opt.dr == 0.1 / 6.0
        assert opt.smooth_rdfs is False
        assert opt.rdf_r_range.shape[0] == 2
        assert opt.pot_r.shape[0] == n_bins

    def test_msibi_init_multiple_cutoff(self):
        opt = MSIBI(2.5, n_bins, pot_cutoff=2.0)
        assert opt.pot_cutoff != opt.rdf_cutoff
        assert opt.n_rdf_points == n_bins
        assert opt.rdf_n_bins == n_bins
        assert opt.r_switch == opt.pot_r[-5]
        assert opt.dr == 0.1 / 6.0
        assert opt.smooth_rdfs is False
        assert opt.rdf_r_range.shape[0] == 2
        assert opt.pot_r.shape[0] != n_bins
        assert opt.pot_r.shape[0] == 121

    def test_msibi_optimize_states(self, state0, pair, tmp_path):
        opt = MSIBI(2.5, n_bins, pot_cutoff=2.5)
        opt.add_state(state0)
        opt.add_pair(pair)
        opt.optimize(
                n_iterations=0,
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=100,
                engine="hoomd",
                _dir=tmp_path)

    def test_rdf_length(self, state0, pair, tmp_path):
        opt = MSIBI(2.5, n_bins + 1, pot_cutoff=2.5)
        opt.add_state(state0)
        opt.add_pair(pair)
        with pytest.raises(ValueError):
            opt.optimize(
                n_iterations=0,
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=100,
                engine="hoomd",
                _dir=tmp_path)
