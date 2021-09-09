import os

import mdtraj as md
import numpy as np
import pytest

from msibi import MSIBI, State, Pair, mie

from .base_test import BaseTest

dr = 0.1 / 6.0
r = np.arange(0, 2.5 + dr, dr)
r_range = np.asarray([0.0, 2.5 + dr])
n_bins = 151
k_B = 1.9872041e-3  # kcal/mol-K
T = 298.0  # K


class TestPair(BaseTest):
    def test_pair_name(self, pair):
        assert pair.name == "0-1"

    def test_save_table_potential(self, tmp_path):
        pair = Pair("A", "B", potential=mie(r, 1.0, 1.0))
        pair.potential_file = os.path.join(tmp_path, "pot.txt")
        pair.save_table_potential(r, dr)
        assert os.path.isfile(pair.potential_file)

    def test_add_state(self, pair, state0, rdf0, tmp_path):
        opt = MSIBI(2.5, n_bins, smooth_rdfs=True, rdf_exclude_bonded=True)
        opt.add_state(state0)
        opt.add_pair(pair)
        opt.optimize(n_iterations=0, _dir=tmp_path)
        assert isinstance(pair._states, dict)
        assert np.array_equal(pair._states[state0]["target_rdf"], rdf0)
        assert pair._states[state0]["current_rdf"] is None
        assert pair._states[state0]["alpha"] == 0.5
        assert pair._states[state0]["pair_indices"] is None
        assert len(pair._states[state0]["f_fit"]) == 0

    def test_calc_current_rdf_no_smooth(self, state0, pair):
        pair.compute_current_rdf(
            state0, r_range, n_bins, smooth=False, max_frames=1e3
        )
        assert pair._states[state0]["current_rdf"] is not None
        assert len(pair._states[state0]["f_fit"]) > 0

    def test_calc_current_rdf_smooth(self, state0, pair):
        pair.compute_current_rdf(
            state0, r_range, n_bins, smooth=True, max_frames=1e3
        )
        assert pair.states[state]["current_rdf"] is not None
        assert len(pair.states[state]["f_fit"]) > 0

    def test_save_current_rdf(self, state0, pair):
        pair.compute_current_rdf(
            state0, r_range, n_bins, smooth=True, max_frames=1e3
        )
        pair.save_current_rdf(state, 0, 0.1 / 6.0)
        if not os.path.isdir("rdfs"):
            os.system("mkdir rdfs")
        assert os.path.isfile("rdfs/pair_0-1-state_state0-step0.txt")

    def test_update_potential(self, state0, pair):
        """Make sure the potential changes after calculating RDF"""
        pair.compute_current_rdf(
            state0, r_range, n_bins, smooth=True, max_frames=1e3
        )
        pair.update_potential(np.arange(0, 2.5 + dr, dr), r_switch=1.8)
        assert not np.array_equal(pair.potential, pair.previous_potential)
