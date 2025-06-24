import os

import numpy as np
import pytest

from msibi import State

from .base_test import BaseTest


class TestState(BaseTest):
    def test_state_init(self, tmp_path, stateX):
        assert stateX.name == "X"
        assert stateX.alpha0 == 1.0
        assert stateX.kT == 1.0
        assert os.path.exists(os.path.join(tmp_path, "states/X_1.0/"))

    def test_n_frames(self, stateX):
        stateX.nframes = 50
        assert stateX.nframes == 50

    def test_alpha0_setter(self, stateX):
        assert stateX.alpha0 == 1.0
        stateX.alpha0 = 0.5
        assert stateX.alpha0 == 0.5
        with pytest.raises(ValueError):
            stateX.alpha0 = -0.5

    def test_constant_alpha_form(self, stateX):
        assert stateX.alpha_form == "constant"
        assert stateX.alpha() == stateX.alpha0

    def test_linear_alpha_form(self, traj_file_path, tmp_path):
        state = State(
            name="X",
            alpha0=1.0,
            kT=1.0,
            traj_file=traj_file_path,
            n_frames=10,
            alpha_form="linear",
            _dir=tmp_path,
        )
        alpha_array = state.alpha(pot_x_range=np.arange(0.1, 2.1, 0.1), dx=0.1)
        assert len(alpha_array) == 20
        assert np.round(alpha_array[-1], 5) == 0
        assert alpha_array[1] != state.alpha0
        assert np.round(alpha_array[0], 5) == state.alpha0

    def test_linear_alpha_form_no_pot(self, traj_file_path, tmp_path):
        state = State(
            name="X",
            alpha0=1.0,
            kT=1.0,
            traj_file=traj_file_path,
            n_frames=10,
            alpha_form="linear",
            _dir=tmp_path,
        )
        with pytest.raises(ValueError):
            state.alpha(pot_x_range=None)

    def test_bad_alpha_form(self, traj_file_path, tmp_path):
        with pytest.raises(ValueError):
            State(
                name="X",
                alpha0=1.0,
                kT=1.0,
                traj_file=traj_file_path,
                n_frames=10,
                alpha_form="exponential",
                _dir=tmp_path,
            )
