import os
import shutil
import warnings
from msibi import MSIBI, utils

import cmeutils as cme
from cmeutils.structure import gsd_rdf
import gsd
import gsd.hoomd
import mdtraj as md


HOOMD2_HEADER = """
import hoomd
import hoomd.md
from hoomd.init import read_gsd

hoomd.context.initialize("")
system = read_gsd("{0}", frame=-1, time_step=0)
T_final = {1:.1f}

pot_width = {2:d}
nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=pot_width, nlist=nl)
harmonic_bond = hoomd.md.bond.harmonic()
harmonic_angle = hoomd.md.angle.harmonic()
"""

HOOMD_TABLE_ENTRY = """
table.set_from_file('{type1}', '{type2}', filename='{potential_file}')
"""

HOOMD_BOND_ENTRY = """
harmonic_bond.bond_coeff.set('{name}', k={k}, r0={r0})
"""

HOOMD_ANGLE_ENTRY = """
harmonic_angle.angle_coeff.set('{name}', k={k}, t0={theta})
"""

class State(object):
    """A single state used as part of a multistate optimization.

    Attributes
    ----------
    kT : float, required
        Unitless heat energy (product of Boltzmann's constant and temperature).
    name : str, required
        State name used in creating state directory space and output files.
    traj_file : path to a gsd.hoomd.HOOMDTrajectory file
        The gsd trajectory associated with this state
    alpha : float, optional, default=1.0
        The alpha value used to scaale the weight of this state.
    backup_trajectory : bool, optional, default=False
        True if each query trajectory is backed up (default False)
    """
    def __init__(
        self,
        name,
        kT,
        traj_file,
        alpha=1.0,
        backup_trajectory=False,
    ):
        self.name = name
        self.kT = kT
        self.traj_file = os.path.abspath(traj_file)
        self._opt = None
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha should be between 0.0 and 1.0")
        self.alpha = float(alpha)
        self.dir = self._setup_dir(name, kT)
        self.query_traj = os.path.join(self.dir, "query.gsd")
        self.backup_trajectory = backup_trajectory
        shutil.copy(
                os.path.join(utils.__path__[0], "hoomd_run_template.py"),
                self.dir
                )

    def reload_query_trajectory(self):
        """Reload the query trajectory. """
        if self.top_path:
            self.traj = md.load(self.traj_file, top=self.top_path)
        else:
            self.traj = md.load(self.traj_file)

    def save_runscript(
        self,
        table_potentials,
        table_width,
        bonds,
        angles,
        engine="hoomd",
        runscript="hoomd_run_template.py",
    ):
        """Save the input script for the MD engine. """
        header = list()
        header.append(
                HOOMD2_HEADER.format(self.traj_file, self.kT, table_width)
                )

        for type1, type2, potential_file in table_potentials:
            header.append(HOOMD_TABLE_ENTRY.format(**locals()))

        if bonds:
            for bond in bonds:
                name = bond.name
                k = bond._states[self]["k"]
                r0 = bond._states[self]["r0"]
                header.append(HOOMD_BOND_ENTRY.format(**locals()))

        if angles:
            for angle in angles:
                name = angle.name
                k = angle._states[self]["k"]
                theta = angle._states[self]["theta"]
                header.append(HOOMD_ANGLE_ENTRY.format(**locals()))

        header = "".join(header)
        with open(os.path.join(self.dir, runscript)) as fh:
            body = "".join(fh.readlines())

        runscript_file = os.path.join(self.dir, "run.py")
        with open(runscript_file, "w") as fh:
            fh.write(header)
            fh.write(body)

    def _setup_dir(self, name, kT):
        """
        Handle the creation of a state specific directory each time a new
        State() object is created.
        """
        if not os.path.isdir("states"):
            os.mkdir("states")

        dir_name = f"{name}_{kT}"
        try:
            assert not os.path.isdir(os.path.join("states", dir_name))
            os.mkdir(os.path.join("states", dir_name))
        except:
            raise AssertionError("A State object has already "+
                            f"been created with a name of {name} "+
                            f"and a kT of {kT}")
        return os.path.abspath(os.path.join("states", dir_name))

