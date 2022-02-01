import pytest

from msibi import MSIBI

from .base_test import BaseTest

n_bins = 151


class TestMSIBI(BaseTest):
    def test_msibi_init(self, state0, pair, tmp_path):
        #TODO REDO THIS TEST
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                max_frames=10,
                n_steps=1e6,
        )
        opt.add_state(state0)
        opt.add_pair(pair)
        opt.optimize_pairs(
                n_iterations=0,
                r_switch=None,
                rdf_exclude_bonded=True,
                smooth_rdfs=False,
                _dir=tmp_path,
            )
        assert opt.dr == 0.1 / 6.0
        assert opt.smooth_rdfs is False
        assert opt.rdf_r_range.shape[0] == 2
        assert opt.pot_r.shape[0] == n_bins

    def test_rdf_length(self, state0, pair, tmp_path):
        #TODO Redo this test
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                max_frames=10,
                n_steps=1e6,
        )
        opt.add_state(state0)
        opt.add_pair(pair)
        with pytest.raises(ValueError):
            opt.optimize_pairs(
                    n_iterations=0,
                    r_switch=None,
                    rdf_exclude_bonded=True,
                    smooth_rdfs=False,
                    _dir=tmp_path,
                )
