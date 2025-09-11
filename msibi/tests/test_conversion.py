import os

import gsd
import pytest

from msibi.utils import conversion

from .base_test import BaseTest

test_assets = os.path.join(os.path.dirname(__file__), "assets")


class TestConversion(BaseTest):
    def test_bad_inputs(self, tmp_path):
        topology = None
        trajectory = os.path.join(test_assets, "lammps.dump")
        output = os.path.join(tmp_path, "lammps.dump.gsd")
        with pytest.raises(RuntimeError):
            conversion.gsd_from_files(topology, trajectory, output=output)

    def test_no_file(self, tmp_path):
        with pytest.raises(RuntimeError):
            output = os.path.join(tmp_path, "lammps.dump.gsd")
            conversion.gsd_from_files("topology.data", "trajectory.dump", output=output)

    def test_lammps_dump(
        self,
        tmp_path,
        topology=os.path.join(test_assets, "lammps.data"),
        trajectory=os.path.join(test_assets, "lammps.dump"),
    ):
        output = os.path.join(tmp_path, "lammps.dump.gsd")
        conversion.gsd_from_files(topology, trajectory, output=output)
        assert os.path.isfile(output)
        with gsd.hoomd.open(output) as traj:
            snap = traj[-1]
            assert len(traj) == 7
            assert snap.particles.N == 9254
            assert snap.bonds.N == 6189
            assert snap.angles.N == 3184
            assert snap.dihedrals.N == 171

    def test_lammps_dcd(
        self,
        tmp_path,
        topology=os.path.join(test_assets, "lammps.data"),
        trajectory=os.path.join(test_assets, "lammps.dcd"),
    ):
        output = os.path.join(tmp_path, "lammps.dcd.gsd")
        conversion.gsd_from_files(topology, trajectory, output=output)
        assert os.path.isfile(output)
        with gsd.hoomd.open(output) as traj:
            snap = traj[-1]
            assert len(traj) == 7
            assert snap.particles.N == 9254
            assert snap.bonds.N == 6189
            assert snap.angles.N == 3184
            assert snap.dihedrals.N == 171

    def test_amber(
        self,
        tmp_path,
        topology=os.path.join(test_assets, "amber.psf"),
        trajectory=os.path.join(test_assets, "amber.nc"),
    ):
        output = os.path.join(tmp_path, "amber.gsd")
        conversion.gsd_from_files(topology, trajectory, output=output)
        assert os.path.isfile(output)
        with gsd.hoomd.open(output) as traj:
            snap = traj[-1]
            assert len(traj) == 12
            assert snap.particles.N == 9254
            assert (
                snap.bonds.N == 9253
            )  # Water has an aditional bond between H - H in charmm_type topologies
            assert snap.angles.N == 3184
            assert snap.dihedrals.N == 171

    def test_gromacs(
        self,
        tmp_path,
        topology=os.path.join(test_assets, "gromacs.tpr"),
        trajectory=os.path.join(test_assets, "gromacs.xtc"),
    ):
        output = os.path.join(tmp_path, "gromacs.gsd")
        conversion.gsd_from_files(topology, trajectory, output=output)
        assert os.path.isfile(output)
        with gsd.hoomd.open(output) as traj:
            snap = traj[-1]
            assert len(traj) == 11
            assert snap.particles.N == 9254
            assert snap.bonds.N == 6189
            assert snap.angles.N == 3184
            assert snap.dihedrals.N == 171
