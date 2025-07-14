import hoomd
import numpy as np
import pytest

from msibi import MSIBI, Angle, Bond, Dihedral, Pair

from .base_test import BaseTest


class TestMSIBI(BaseTest):
    def test_init(self, msibi):
        assert msibi.n_iterations == 0
        assert isinstance(msibi.nlist(buffer=0.20), hoomd.md.nlist.Cell)
        assert isinstance(
            msibi.integrator_method(filter=hoomd.filter.All()),
            hoomd.md.methods.ConstantVolume,
        )
        assert isinstance(
            msibi.thermostat(kT=1.0, tau=0.01),
            hoomd.md.methods.thermostats.MTTK,
        )

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
        bond.set_polynomial(x_min=0.0, x_max=3.0, x0=1, k2=200, k3=0, k4=0)
        msibi.add_state(stateX)
        msibi.add_state(stateY)
        msibi.add_force(bond)
        init_bond_pot = np.copy(bond.potential)
        msibi.run_optimization(n_steps=500, n_iterations=1)
        assert not np.array_equal(bond.potential, init_bond_pot)
        assert msibi.n_iterations == 1
        ff = msibi._build_force_objects()
        assert len(ff) == 1
        assert len(bond.distribution_history(state=stateX)) == 1
        assert len(bond.potential_history) == 2
        assert len(bond._head_correction_history) == 1
        assert len(bond._tail_correction_history) == 1
        assert len(bond._learned_potential_history) == 1

    def test_run_with_static_force(self, msibi, stateX, stateY):
        msibi.gsd_period = 10
        bond = Bond(type1="A", type2="B", optimize=True, nbins=60)
        bond.set_polynomial(x_min=0.0, x_max=3.0, x0=1, k2=200, k3=0, k4=0)
        angle = Angle(type1="A", type2="B", type3="A", optimize=False)
        angle.set_harmonic(t0=1.9, k=100)
        angle2 = Angle(type1="B", type2="A", type3="B", optimize=False)
        angle2.set_harmonic(t0=2.3, k=100)
        msibi.add_force(angle)
        msibi.add_force(angle2)
        msibi.add_state(stateX)
        msibi.add_state(stateY)
        msibi.add_force(bond)
        init_bond_pot = np.copy(bond.potential)
        msibi.run_optimization(n_steps=500, n_iterations=1)
        assert not np.array_equal(bond.potential, init_bond_pot)
        assert msibi.n_iterations == 1
        ff = msibi._build_force_objects()
        assert len(ff) == 2

    def test_optimize_pairs_linear_alpha(
        self, msibi, stateX_linear_alpha, stateY
    ):
        msibi.gsd_period = 10
        msibi.add_state(stateX_linear_alpha)
        msibi.add_state(stateY)

        bond = Bond(type1="A", type2="B", optimize=False, nbins=60)
        bond.set_harmonic(r0=1.1, k=100)
        msibi.add_force(bond)

        angle = Angle(type1="A", type2="B", type3="A", optimize=False)
        angle.set_harmonic(t0=1.9, k=100)
        msibi.add_force(angle)

        angle2 = Angle(type1="B", type2="A", type3="B", optimize=False)
        angle2.set_harmonic(t0=2.3, k=100)
        msibi.add_force(angle2)

        pair = Pair(
            type1="A",
            type2="B",
            r_cut=2.0,
            nbins=100,
            optimize=True,
            exclude_bonded=True,
        )
        pair.set_lj(sigma=1.5, epsilon=1, r_cut=2.0, r_min=0.1)
        msibi.add_force(pair)

        pair2 = Pair(
            type1="A",
            type2="A",
            r_cut=2.0,
            nbins=100,
            optimize=True,
            exclude_bonded=True,
        )
        pair2.set_lj(sigma=2, epsilon=2, r_cut=2.0, r_min=0.1)
        msibi.add_force(pair2)

        pair3 = Pair(
            type1="B",
            type2="B",
            r_cut=2.0,
            nbins=100,
            optimize=True,
            exclude_bonded=True,
        )
        pair3.set_lj(sigma=1.5, epsilon=1, r_cut=2.0, r_min=0.1)
        msibi.add_force(pair3)

        msibi.run_optimization(n_steps=500, n_iterations=1)

    def test_run_with_all_forces(self, msibi, stateX, stateY):
        msibi.gsd_period = 10
        msibi.add_state(stateX)
        msibi.add_state(stateY)

        bond = Bond(type1="A", type2="B", optimize=False, nbins=60)
        bond.set_harmonic(r0=1.1, k=100)
        msibi.add_force(bond)

        angle = Angle(type1="A", type2="B", type3="A", optimize=False)
        angle.set_harmonic(t0=1.9, k=100)
        msibi.add_force(angle)

        angle2 = Angle(type1="B", type2="A", type3="B", optimize=False)
        angle2.set_harmonic(t0=2.3, k=100)
        msibi.add_force(angle2)

        pair = Pair(
            type1="A",
            type2="B",
            r_cut=2.0,
            nbins=100,
            optimize=True,
            exclude_bonded=True,
        )
        pair.set_lj(sigma=1.5, epsilon=1, r_cut=2.0, r_min=0.1)
        msibi.add_force(pair)

        pair2 = Pair(
            type1="A",
            type2="A",
            r_cut=2.0,
            nbins=100,
            optimize=True,
            exclude_bonded=True,
        )
        pair2.set_lj(sigma=2, epsilon=2, r_cut=2.0, r_min=0.1)
        msibi.add_force(pair2)

        pair3 = Pair(
            type1="B",
            type2="B",
            r_cut=2.0,
            nbins=100,
            optimize=True,
            exclude_bonded=True,
        )
        pair3.set_lj(sigma=1.5, epsilon=1, r_cut=2.0, r_min=0.1)
        msibi.add_force(pair3)

        dihedral = Dihedral(
            type1="B", type2="A", type3="B", type4="A", optimize=False
        )
        dihedral.set_periodic(k=100, phi0=0, d=-1, n=1)
        msibi.add_force(dihedral)

        msibi.run_optimization(n_steps=500, n_iterations=1)

    def test_raise_errors(self, msibi, stateX, stateY):
        with pytest.raises(RuntimeError):
            msibi.pickle_forces(file_path="test.pkl")

        bond = Bond(type1="A", type2="B", optimize=True, nbins=60)
        angle = Angle(type1="A", type2="B", type3="A", optimize=True, nbins=60)
        with pytest.raises(RuntimeError):
            msibi.add_force(bond)
            msibi.add_force(angle)

        with pytest.raises(ValueError):
            msibi = MSIBI(
                nlist=hoomd.md.nlist.Cell,
                integrator_method=hoomd.md.methods.DisplacementCapped,
                method_kwargs=dict(),
                thermostat=hoomd.md.methods.thermostats.MTTK,
                thermostat_kwargs=dict(tau=0.01),
                dt=0.003,
                gsd_period=int(1e3),
            )
