import os
import shutil

import cmeutils as cme
from cmeutils.structure import gsd_rdf
import gsd
import gsd.hoomd
import mdtraj as md
import numpy as np

HOOMD2_HEADER = """
import hoomd
import hoomd.md
from hoomd.init import read_gsd

hoomd.context.initialize("")
try:
    system = read_gsd("{0}", frame=-1, time_step=0)
except RuntimeError:
    from hoomd.deprecated.init import read_xml
    system = read_xml(filename="{0}", wrap_coordinates=True)
T_final = {1:.1f}

pot_width = {2:d}
nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=pot_width, nlist=nl)

"""

HOOMD_TABLE_ENTRY = """
table.set_from_file('{type1}', '{type2}', filename='{potential_file}')
"""


class State(object):
    """A single state used as part of a multistate optimization.

    Attributes
    ----------
    kT : float
        Unitless heat energy (product of Boltzmann's constant and temperature).
    target_rdf : numpy array
        Target RDF obtained from atomistic/ua system
    name : str
        State name. If no name is given, state will be named 'state-{kT:.3f}'
        (default None)

    state_dir : path
        Path to state directory (default '')
    traj_file : path or md.Trajectory
        The dcd or gsd trajectory associated with this state
        (default 'query.dcd')
    top_file : path
        hoomdxml containing topology information (needed for dcd)
        (default None)
    backup_trajectory : bool
        True if each query trajectory is backed up (default False)

    """

    def __init__(
        self,
        name,
        kT,
        traj_file,
        target_rdf=None,
        top_file=None,
        backup_trajectory=False,
    ):
        
        self.name = name
        self.kT = kT
        self.dir = self._setup_dir(name, kT) 
        self.traj_file = traj_file
        self.traj = None
        self.backup_trajectory = backup_trajectory
        self.target_rdf = os.path.join(self.dir, "target_rdf.txt")
        
        if target_rdf is not None:
            # Target RDF passed into State class in the form of a txt file
            # Copy to self.dir, rename to target_rdf.txt
            if isinstance(target_rdf, str):
                shutil.copyfile(target_rdf,
                        os.path.join(self.dir, "target_rdf.txt")
                        )
        
            # Target RDF passed in State class in the form of an array of data
            # Save in self.dir, name target_rdf.txt
            elif isinstance(target_rdf, np.ndarray):
                np.savetxt(os.path.join(self.dir, "target_rdf.txt"), target_rdf)

    
    def target_rdf_calc(
            self,
            A_name,
            B_name,
            exclude_bonded=False
        ):
        """
        Calculate and store the RDF data from a trajectory file of a particular
        State. 
        """        
        rdf, norm = gsd_rdf(
                self.traj_file,
                A_name,
                B_name,
                # TODO: These 3 come from MSIBI()
                start=-5, 
                r_max=4, 
                bins=100,
                exclude_bonded=exclude_bonded
        )
        data = np.stack((rdf.bin_centers, rdf.rdf*norm)).T
        np.savetxt(os.path.join(self.dir, "target_rdf.txt"), data)


    def reload_query_trajectory(self):
        """Reload the query trajectory. """
        if self.top_path:
            self.traj = md.load(self.traj_path, top=self.top_path)
        else:
            self.traj = md.load(self.traj_path)

    def save_runscript(
        self,
        table_potentials,
        table_width,
        engine="hoomd",
        runscript="hoomd_run_template.py",
    ):
        """Save the input script for the MD engine. """

        header = list()

        if self.HOOMD_VERSION == 1:
            HOOMD_HEADER = HOOMD1_HEADER
        elif self.HOOMD_VERSION == 2:
            HOOMD_HEADER = HOOMD2_HEADER

        if self._is_gsd:
            header.append(
                    HOOMD_HEADER.format(self.traj_file, self.kT, table_width)
                    )
        else:
            header.append(
                    HOOMD_HEADER.format(self.top_path, self.kT, table_width)
                    )
        for type1, type2, potential_file in table_potentials:
            header.append(HOOMD_TABLE_ENTRY.format(**locals()))
        header = "".join(header)
        with open(os.path.join(self.state_dir, runscript)) as fh:
            body = "".join(fh.readlines())

        runscript_file = os.path.join(self.state_dir, "run.py")
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








