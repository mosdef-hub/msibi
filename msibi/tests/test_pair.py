import os

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
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                n_potential_points=n_bins,
                potential_cutoff=2.5,
                r_min=1e-4,
                max_frames=10,
                n_iterations=0,
                n_steps=1e6,
        )
        opt.add_state(state0)
        opt.add_pair(pair)
        opt.optimize_pairs(
                r_switch=None,
                rdf_exclude_bonded=True,
                smooth_rdfs=True,
                _dir=tmp_path,
            )
        assert isinstance(pair._states, dict)
        assert np.array_equal(pair._states[state0]["target_rdf"], rdf0)
        assert pair._states[state0]["current_rdf"] is None
        assert pair._states[state0]["alpha"] == 0.5
        assert pair._states[state0]["pair_indices"] is None
        assert len(pair._states[state0]["f_fit"]) == 0

    def test_current_rdf_no_smooth(self, state0, pair, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                potential_cutoff=2.5,
                n_potential_points=n_bins,
                r_min=1e-4,
                n_steps=1e6,
                n_iterations=0,
                max_frames=10
        )
        opt.add_state(state0)
        opt.add_pair(pair)
        opt.optimize_pairs(
                r_switch=None,
                rdf_exclude_bonded=True,
                smooth_rdfs=False,
                _dir=tmp_path,
            )
        pair.compute_current_rdf(state0, opt.smooth_rdfs, query=False)
        assert pair._states[state0]["current_rdf"] is not None
        assert len(pair._states[state0]["f_fit"]) > 0

    def test_current_rdf_smooth(self, state0, pair, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                potential_cutoff=2.5,
                n_potential_points=n_bins,
                r_min=1e-4,
                n_steps=1e6,
                n_iterations=0,
                max_frames=10
        )
        opt.add_state(state0)
        opt.add_pair(pair)
        opt.optimize_pairs(
                r_switch=None,
                rdf_exclude_bonded=True,
                smooth_rdfs=True,
                _dir=tmp_path,
            )
        pair.compute_current_rdf(state0, opt.smooth_rdfs, query=False)
        assert pair._states[state0]["current_rdf"] is not None
        assert len(pair._states[state0]["f_fit"]) > 0

    def test_save_current_rdf(self, state0, pair, tmp_path):
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                potential_cutoff=2.5,
                n_potential_points=n_bins,
                r_min=1e-4,
                n_steps=1e6,
                n_iterations=0,
                max_frames=10
        )
        opt.add_state(state0)
        opt.add_pair(pair)
        opt.optimize_pairs(
                r_switch=None,
                rdf_exclude_bonded=True,
                smooth_rdfs=True,
                _dir=tmp_path,
            )
        pair.compute_current_rdf(state0, opt.smooth_rdfs, query=False)
        pair.save_current_rdf(state0, 0, opt.dr)
        assert os.path.isfile(
            os.path.join(
                state0.dir, f"pair_{pair.name}-state_{state0.name}-step0.txt"
            )
        )

    def test_update_potential(self, state0, pair, tmp_path):
        """Make sure the potential changes after calculating RDF"""
        opt = MSIBI(
                integrator="hoomd.md.integrate.nvt",
                integrator_kwargs={"tau": 0.1},
                dt=0.001,
                gsd_period=1000,
                n_potential_points=n_bins,
                potential_cutoff=2.5,
                r_min=1e-4,
                max_frames=10,
                n_iterations=0,
                n_steps=1e6,
        )
        opt.add_state(state0)
        opt.add_pair(pair)
        opt.optimize_pairs(
                r_switch=None,
                rdf_exclude_bonded=True,
                smooth_rdfs=True,
                _dir=tmp_path,
            )
        pair.compute_current_rdf(state0, opt.smooth_rdfs, query=False)
        pair.update_potential(np.arange(0, 2.5 + dr, dr), r_switch=1.8)
        assert not np.array_equal(pair.potential, pair.previous_potential)
