import pytest

from msibi import MSIBI

from .base_test import BaseTest

n_bins = 151


class TestMSIBI(BaseTest):
    def test_msibi_init(self, state0, pair, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                n_iterations=0,
                n_steps=1e6,
        )
        opt.add_state(state0)
        opt.add_pair(pair)
        opt.optimize_pairs(
                max_frames=10,
                rdf_cutoff=2.5,
                pot_cutoff=None,
                r_min=1e-4,
                r_switch=None,
                n_rdf_points=n_bins,
                rdf_exclude_bonded=True,
                smooth_rdfs=False,
                _dir=tmp_path,
            )
        assert opt.pot_cutoff == opt.rdf_cutoff
        assert opt.n_rdf_points == n_bins
        assert opt.rdf_n_bins == n_bins
        assert opt.r_switch == opt.pot_r[-5]
        assert opt.dr == 0.1 / 6.0
        assert opt.smooth_rdfs is False
        assert opt.rdf_r_range.shape[0] == 2
        assert opt.pot_r.shape[0] == n_bins

    def test_rdf_length(self, state0, pair, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                n_iterations=0,
                n_steps=1e6,
        )
        opt.add_state(state0)
        opt.add_pair(pair)
        with pytest.raises(ValueError):
            opt.optimize_pairs(
                    max_frames=10,
                    rdf_cutoff=2.5,
                    pot_cutoff=None,
                    r_min=1e-4,
                    r_switch=None,
                    n_rdf_points=n_bins+1,
                    rdf_exclude_bonded=True,
                    smooth_rdfs=False,
                    _dir=tmp_path,
                )
