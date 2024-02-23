import os

import numpy as np
import pytest

from msibi import MSIBI, State, Pair
from msibi.potentials import save_table_potential

from .base_test import BaseTest

dr = 0.1 / 6.0
r = np.arange(0, 2.5 + dr, dr)
r_range = np.asarray([0.0, 2.5 + dr])
n_bins = 151
k_B = 1.9872041e-3  # kcal/mol-K
T = 298.0  # K


class TestPair(BaseTest):
    def test_pair_name(self, pairs):
        assert pairs[0].name == "0-0"

    def test_save_table_potential(self, tmp_path):
        pair = Pair("0", "1")
        pair.set_table_potential(1, 1, 0, 2.5, 100)
        pair.potential_file = os.path.join(tmp_path, "pot.txt")
        save_table_potential(
                pair.potential,
                pair.r_range,
                pair.dr,
                None,
                pair.potential_file
        )
        assert os.path.isfile(pair.potential_file)

    def test_add_state(self, pairs, state0, rdf0, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                nlist="hoomd.md.nlist.cell",
                dt=0.001,
                gsd_period=1000,
                n_steps=1e6,
        )
        opt.add_state(state0)
        opt.add_pair(pairs[3])
        opt.optimize_pairs(
                n_iterations=0,
                r_switch=None,
                smooth_rdfs=False,
                _dir=tmp_path,
        )
        assert isinstance(pairs[0]._states, dict)
        #assert np.array_equal(pairs[3]._states[state0]["target_distribution"], rdf0)
        assert pairs[3]._states[state0]["current_distribution"] is None
        assert pairs[3]._states[state0]["alpha"] == 0.5
        assert len(pairs[3]._states[state0]["f_fit"]) == 0
    
    @pytest.mark.skip(reason="Need better test GSDs before running IBI in tests")
    def test_current_rdf_no_smooth(self, state0, pairs, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                nlist="hoomd.md.nlist.cell",
                dt=0.001,
                gsd_period=5000,
                n_steps=1e4,
        )
        opt.add_state(state0)
        for pair in pairs:
            opt.add_pair(pair)
        opt.optimize_pairs(
                n_iterations=1,
                r_switch=None,
                smooth_rdfs=False,
                _dir=tmp_path,
            )
        pairs[3]._compute_current_rdf(state0)
        assert pairs[3]._states[state0]["current_distribution"] is not None
        assert len(pairs[3]._states[state0]["f_fit"]) > 0

    @pytest.mark.skip(reason="Need better test GSDs before running IBI in tests")
    def test_current_rdf_smooth(self, state0, pairs, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                nlist="hoomd.md.nlist.cell",
                dt=0.001,
                gsd_period=5000,
                n_steps=1e4,
        )
        opt.add_state(state0)
        for pair in pairs:
            opt.add_pair(pair)
        opt.optimize_pairs(
                n_iterations=1,
                r_switch=None,
                smooth_rdfs=True,
                _dir=tmp_path,
            )
        pairs[3]._compute_current_rdf(state0)
        assert pairs[3]._states[state0]["current_distribution"] is not None
        assert len(pairs[3]._states[state0]["f_fit"]) > 0

    def test_save_current_rdf(self, state0, pairs, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                nlist="hoomd.md.nlist.cell",
                dt=0.001,
                gsd_period=1000,
                n_steps=1e6,
        )
        opt.add_state(state0)
        for pair in pairs:
            opt.add_pair(pair)
        opt.optimize_pairs(
                n_iterations=0,
                r_switch=None,
                smooth_rdfs=True,
                _dir=tmp_path,
            )
        target_rdf = pairs[0]._states[state0]["target_distribution"]
        pairs[0]._states[state0]["current_distribution"] = target_rdf 
        pairs[0]._save_current_rdf(state0, 0)
        assert os.path.isfile(
            os.path.join(
                state0.dir, f"pair_rdf_{pairs[0].name}-state_{state0.name}-step0.txt"
            )
        )

    def test_update_potential(self, state0, pairs, tmp_path):
        """Make sure the potential changes after calculating RDF"""
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                nlist="hoomd.md.nlist.cell",
                dt=0.001,
                gsd_period=1000,
                n_steps=1e6,
        )
        opt.add_state(state0)
        for pair in pairs:
            opt.add_pair(pair)
        opt.optimize_pairs(
                n_iterations=0,
                r_switch=None,
                smooth_rdfs=False,
                _dir=tmp_path,
            )
        target_rdf = pairs[1]._states[state0]["target_distribution"]
        pairs[0]._states[state0]["current_distribution"] = target_rdf
        pairs[0]._update_potential(smooth=False, smoothing_window=5)
        assert not np.array_equal(pairs[0].potential, pairs[0].previous_potential)
