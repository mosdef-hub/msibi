import pytest

from msibi import MSIBI

from .base_test import BaseTest

n_bins = 151


class TestMSIBI(BaseTest):
    def test_add_potential_objects(self, state0, pair, bond, angle):
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
        opt.add_bond(bond)
        opt.add_angle(angle)
        assert len(opt.pairs) == len(opt.bonds) == len(opt.angles) == 1

    def test_opt_pairs(self, state0, pair, tmp_path):
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
        assert opt.optimization == "pairs"
        assert pair.r_switch == pair.r_range[-5]
        for key in pair._states.keys():
            assert key == state0

    def test_opt_bonds(self, state0, bond, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                max_frames=10,
                n_steps=1e6,
        )
        opt.add_state(state0)
        opt.add_bond(bond)
        opt.optimize_bonds(
                n_iterations=0,
            )
        assert opt.optimization == "bonds"
        for key in bond._states.keys():
            assert key == state0

    def test_opt_angles(self, state0, angle, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                max_frames=10,
                n_steps=1e6,
        )
        opt.add_state(state0)
        opt.add_angle(angle)
        opt.optimize_angles(
                n_iterations=0,
            )
        assert opt.optimization == "angles"
        for key in angle._states.keys():
            assert key == state0

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
