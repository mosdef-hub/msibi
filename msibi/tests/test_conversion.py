import os
import pytest
from msibi.utils import conv
import gsd
from .base_test import BaseTest

test_assets = os.path.join(os.path.dirname(__file__), "assets")

class TestConversion(BaseTest):
    def test_lammps_dump(self, tmp_path, 
                        topology=os.path.join(test_assets, 'lammps.data'), 
                        trajectory=os.path.join(test_assets, 'lammps.dump')):
        output=os.path.join(tmp_path,'lammps.dump.gsd')
        conv.gsd_from_files(topology,trajectory, output=output)
        assert os.path.isfile(output)
        with gsd.hoomd.open(output) as traj:
            snap = traj[-1]
            assert snap.particles.N == 9254
            assert snap.bonds.N     == 6189
            assert snap.angles.N    == 3184
            assert snap.dihedrals.N == 171
    def test_lammps_dcd(self, tmp_path, 
                        topology=os.path.join(test_assets, 'lammps.data'), 
                        trajectory=os.path.join(test_assets, 'lammps.dcd')):
        output=os.path.join(tmp_path,'lammps.dcd.gsd')
        conv.gsd_from_files(topology,trajectory, output=output)
        assert os.path.isfile(output)
        with gsd.hoomd.open(output) as traj:
            snap = traj[-1]
            assert snap.particles.N == 9254
            assert snap.bonds.N     == 6189
            assert snap.angles.N    == 3184
            assert snap.dihedrals.N == 171
    def test_amber(self, tmp_path, 
                        topology=os.path.join(test_assets, 'amber.psf'), 
                        trajectory=os.path.join(test_assets, 'amber.nc')):
        output=os.path.join(tmp_path,'amber.gsd')
        conv.gsd_from_files(topology,trajectory, output=output)
        assert os.path.isfile(output)
        with gsd.hoomd.open(output) as traj:
            snap = traj[-1]
            assert snap.particles.N == 9254
            assert snap.bonds.N     == 9253 #Water has an aditional bond between H - H in charmm_type topologies 
            assert snap.angles.N    == 3184
            assert snap.dihedrals.N == 171
    def test_gromacs(self, tmp_path, 
                        topology=os.path.join(test_assets, 'gromacs.tpr'), 
                        trajectory=os.path.join(test_assets, 'gromacs.xtc')):
        output=os.path.join(tmp_path,'gromacs.gsd')
        conv.gsd_from_files(topology,trajectory, output=output)
        assert os.path.isfile(output)
        with gsd.hoomd.open(output) as traj:
            snap = traj[-1]
            assert snap.particles.N == 9254
            assert snap.bonds.N     == 6189
            assert snap.angles.N    == 3184
            assert snap.dihedrals.N == 171