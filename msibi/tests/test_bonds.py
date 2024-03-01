import os

import numpy as np
import pytest

from msibi import MSIBI, State, Bond, Angle 
from msibi.potentials import save_table_potential

from .base_test import BaseTest


class TestBond(BaseTest):
    def test_bond_name(self, bond=bondAB(optimize=False)):
        assert bond.name == "A-B"
        assert bond.optimize is False

    def test_set_harmonic(self bond=bondAB(optimize=False)):
        bond.set_harmonic(k=500, l0=2)
        assert bond.format == "static"
        assert bond.force_entry["k"] == 500
        assert bond.force_entry["l0"] == 2 

    def test_set_quadratic(self, bond=bondAB(optimize=True)):
        assert bond.optimize is True
        bond.set_quadratic(
                x0=2,
                k4=1,
                k3=1,
                k2=1,
                xmin=1,
                xmax=3,
        )
        assert bond.bond_type == "table"
        assert bond.x_range[0] == 1
        assert bond.x_range[-1] == 3
        assert len(bond.potential) == bond.nbins + 1 

    def test_save_table_potential(self, tmp_path, bond=bondAB(optimize=True)):
        bond.set_quadratic(
                x0=2,
                k4=1,
                k3=1,
                k2=1,
                xmin=1,
                xmax=3,
        )
        path = os.path.join(tmp_path, "AB_bond.csv")
        bond.save_potential(path)
        assert os.path.isfile(path)

class TestAngle(BaseTest):
    def test_angle_name(self, angle=angleABA(optimize=False)):
        assert angle.name == "A-B-A"
        assert angle.optimize is False

    def test_set_angle_harmonic(self):
        angle.set_harmonic(k=500, theta0=2)
        assert angle.angle_type == "static"
        assert angle.force_entry["t0"] = 2
        assert angle.force_entry["k"] = 500 

    def test_set_quadratic(self, angle=angleABA(optimize=True)):
        angle.set_quadratic(
                theta0=2,
                k4=0,
                k3=0,
                k2=100,
        )
        assert angle.format == "table"
        assert len(angle.x_range) == angle.nbins + 1

    def test_save_angle_potential(
            self,
            tmp_path,
            angle=angleABA(optimize=True)
    ):
        angle.set_quadratic(
                theta0=2,
                k4=0,
                k3=0,
                k2=100,
        )
        path = os.path.join(tmp_path, "ABA_angle.csv")
        angle.save_potential(path)
        assert os.path.isfile(path)

class TestPair(BaseTest):
    def test_pair_name(self, pair=pairAB(optimize=False)):
        assert pair.name == "A-B"
        assert pair._pair_name = ("A", "B")
        assert pair.optimize is False

    def test_set_lj(self, pair=pairAB(optimize=True)):
        angle.set_quadratic(
                theta0=2,
                k4=0,
                k3=0,
                k2=100,
        )
        assert pair.format == "table"

    def test_save_angle_potential(
            self,
            tmp_path,
            angle=angleABA(optimize=True)
    ):
        angle.set_quadratic(
                theta0=2,
                k4=0,
                k3=0,
                k2=100,
        )
        path = os.path.join(tmp_path, "ABA_angle.csv")
        angle.save_potential(path)
        assert os.path.isfile(path)
