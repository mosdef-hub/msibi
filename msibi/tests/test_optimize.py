import pytest
import hoomd
from msibi import MSIBI, Bond, Angle, Dihedral, Pair

from .base_test import BaseTest



class TestMSIBI(BaseTest):
    def test_init(self, msibi):
        assert msibi.n_iterations == 0
        assert isinstance(msibi.nlist(buffer=0.20), hoomd.md.nlist.Cell)
        assert isinstance(msibi.integrator_method(filter=hoomd.filter.All()), hoomd.md.methods.ConstantVolume)
        assert isinstance(msibi.thermostat(kT=1.0, tau=0.01), hoomd.md.methods.thermostats.MTTK)

    def test_add_state(self, msibi, stateX, stateY):
        msibi.add_state(stateX)
        msibi.add_state(stateY)
        assert msibi.states[0] == stateX
        assert msibi.states[1] == stateY
        assert len(msibi.states) == 2

    def test_add_forces(self, msibi, pairA, bond, angle, dihedral):
        msibi.add_force(pairA)
        msibi.add_force(bond)
        msibi.add_force(angle)
        msibi.add_force(dihedral)
        assert msibi.forces[0] == pairA
        assert msibi.forces[1] == bond
        assert msibi.forces[2] == angle
        assert msibi.forces[3] == dihedral
        assert len(msibi.forces) == 4
        assert len(msibi.pairs) == 1
        assert len(msibi.bonds) == 1
        assert len(msibi.angles) == 1
        assert len(msibi.dihedrals) == 1

    def test_run(self, msibi, stateX, stateY):
        msibi.gsd_period = 10
        bond = Bond(type1="A", type2="B", optimize=True, nbins=60)
        bond.set_quadratic(x_min=0.0, x_max=3.0, x0=1, k2=200, k3=0, k4=0)
        msibi.add_state(stateX)
        msibi.add_state(stateY)
        msibi.add_force(bond)
        msibi.run_optimization(n_steps=500, n_iterations=1)

    


