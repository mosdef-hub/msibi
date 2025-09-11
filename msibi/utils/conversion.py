from typing import Union
import warnings

import gsd.hoomd
import MDAnalysis as mda
import numpy as np
import MDAnalysis
from MDAnalysis.exceptions import NoDataError


def gsd_from_files(topology_file: str, traj_file: str, output: str = "output.gsd"):
    """Convert a topology and trajectory pair to a GSD file after validation.

    This function first validates the inputs using `_mda_check`. If all
    requirements are satisfied, the data is converted into HOOMD's
    GSD format using `gsd_from_universe`.

    .. note::

        If requirements for loading files into MDAnalysis are not met,
        diagnostic information is printed instead of writing a file.

    Examples::

        from msibi.utils.conversion import gsd_from_files

        gsd_from_files(topology_file="lammps.data", traj_file="lammps.dump", output="output.gsd")

    Parameters
    ----------
    topology_file : str
        Path to topology file (PSF, GRO, LAMMPSDATA, etc.).
    traj_file : str
        Path to trajectory file (DCD, XTC, TRR, LAMMPSDUMP etc.).
    output : str, optional, default = output.gsd
        Name of the output GSD file.
    """
    out = _mda_check(topology_file, traj_file)
    ok = _requirements_met(out)

    if ok:
        try:
            gsd_from_universe(mda.Universe(topology_file, traj_file), output)
        except TypeError:
            gsd_from_universe(
                mda.Universe(topology_file, traj_file, format="LAMMPSDUMP"), output
            )
    else:
        msg = f"Checks failed! See the summary of the results: {out}"
        raise RuntimeError(msg)


def gsd_from_universe(universe: MDAnalysis.Universe, output:str="output.gsd"):
    """Convert an MDAnalysis Universe into a GSD trajectory.

    Extracts particle, bonding, angular, dihedral, and improper information
    from the given Universe and writes each frame to a GSD file compatible
    with HOOMD-blue.

    .. note::

        Particle types are inferred from `atom.type`. Bond, angle, dihedral and improper types
        are deduplicated and stored as HOOMD-Blue style type lists with associated type IDs.
        Each frame of the Universe trajectory is written to the GSD file.

    Examples::

        import MDAnalysis as mda
        from msibi.utils.conversion import gsd_from_universe

        u = mda.Universe("topology.tpr", "trajectory.xtc")
        gsd_from_universe(u, output="output.gsd")

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe containing the system's topology and trajectory.
    output : str, optional
        Name of the output GSD file (default is 'output.gsd').

    """
    # Load structure and trajectory
    u = universe

    # Extract atom types
    types = [atom.type for atom in u.atoms]
    unique_types = sorted(set(types))
    typeid = np.array([unique_types.index(t) for t in types], dtype=np.uint32)

    # Extract bonds
    bonds = [(b[0].index, b[1].index) for b in u.bonds]
    bond_types = []
    bond_type_ids = []
    bond_lookup = {}

    for a1, a2 in bonds:
        # bond_type = f"{u.atoms[a1].type}-{u.atoms[a2].type}"
        n1, n2 = u.atoms[a1].type, u.atoms[a2].type
        bond_type = "-".join(sorted((n1, n2)))
        if bond_type not in bond_lookup:
            bond_lookup[bond_type] = len(bond_types)
            bond_types.append(bond_type)
        bond_type_ids.append(bond_lookup[bond_type])

    # Extract angles
    angles = []
    angle_types = []
    angle_type_ids = []
    angle_lookup = {}

    if hasattr(u, "angles") and len(u.angles) > 0:
        for a in u.angles:
            angles.append((a[0].index, a[1].index, a[2].index))
            angle_type = f"{u.atoms[a[0].index].type}-{u.atoms[a[1].index].type}-{u.atoms[a[2].index].type}"
            if angle_type not in angle_lookup:
                angle_lookup[angle_type] = len(angle_types)
                angle_types.append(angle_type)
            angle_type_ids.append(angle_lookup[angle_type])

    # Extract dihedrals
    dihedrals = []
    dihedral_types = []
    dihedral_type_ids = []
    dihedral_lookup = {}

    if hasattr(u, "dihedrals") and len(u.dihedrals) > 0:
        for d in u.dihedrals:
            dihedrals.append((d[0].index, d[1].index, d[2].index, d[3].index))
            dihedral_type = (
                f"{u.atoms[d[0].index].type}-"
                f"{u.atoms[d[1].index].type}-"
                f"{u.atoms[d[2].index].type}-"
                f"{u.atoms[d[3].index].type}"
            )
            if dihedral_type not in dihedral_lookup:
                dihedral_lookup[dihedral_type] = len(dihedral_types)
                dihedral_types.append(dihedral_type)
            dihedral_type_ids.append(dihedral_lookup[dihedral_type])

    # Extract impropers
    impropers = []
    improper_types = []
    improper_lookup = {}
    improper_type_ids = []
    if hasattr(u, "impropers") and len(u.impropers) > 0:
        for imp in u.impropers:
            impropers.append((imp[0].index, imp[1].index, imp[2].index, imp[3].index))
            improper_type = (
                f"{u.atoms[imp[0].index].type}-"
                f"{u.atoms[imp[1].index].type}-"
                f"{u.atoms[imp[2].index].type}-"
                f"{u.atoms[imp[3].index].type}"
            )

            if improper_type not in improper_lookup:
                improper_lookup[improper_type] = len(improper_types)
                improper_types.append(improper_type)

            improper_type_ids.append(improper_lookup[improper_type])

    # Write to GSD
    with gsd.hoomd.open(output, "w") as gsd_file:
        for ts in u.trajectory:
            snap = gsd.hoomd.Frame()

            # Box
            box = ts.dimensions
            snap.configuration.box = [box[0], box[1], box[2], 0, 0, 0]

            # Particles
            snap.particles.N = u.atoms.n_atoms
            snap.particles.position = u.atoms.positions.astype(np.float32)
            snap.particles.typeid = typeid
            snap.particles.types = unique_types

            # Bonds
            if bonds:
                snap.bonds.N = len(bonds)
                snap.bonds.group = np.array(bonds, dtype=np.int32)
                snap.bonds.typeid = np.array(bond_type_ids, dtype=np.uint32)
                snap.bonds.types = bond_types

            # Angles
            if angles:
                snap.angles.N = len(angles)
                snap.angles.group = np.array(angles, dtype=np.int32)
                snap.angles.typeid = np.array(angle_type_ids, dtype=np.uint32)
                snap.angles.types = angle_types

            # Dihedrals
            if dihedrals:
                snap.dihedrals.N = len(dihedrals)
                snap.dihedrals.group = np.array(dihedrals, dtype=np.int32)
                snap.dihedrals.typeid = np.array(dihedral_type_ids, dtype=np.uint32)
                snap.dihedrals.types = dihedral_types
            # Imropers
            if impropers:
                snap.impropers.N = len(impropers)
                snap.impropers.group = np.array(impropers, dtype=np.int32)
                snap.impropers.typeid = np.array(improper_type_ids, dtype=np.uint32)
                snap.impropers.types = improper_types

            gsd_file.append(snap)

    with gsd.hoomd.open(output) as traj:
        snap = traj[0]
        print(f"#Particles:  {snap.particles.N}")
        print(f"#Bonds:      {snap.bonds.N}")
        print(f"#Angles:     {snap.angles.N}")
        print(f"#Dihedrals:  {snap.dihedrals.N}")
        print(
            "\nDouble-check the number of Bonds, Angles, and Dihedrals.\n"
            "Especially when using *.tpr topologies and SETTLE constraints.\n"
        )


def _mda_check(topology: str, trajectory):
    """Validate an MDAnalysis topology/trajectory pair for conversion.

    This function attempts to load the provided topology and trajectory files
    using MDAnalysis and checks whether all structural requirements
    (bonds, angles, etc.) are satisfied, depending on residue sizes.

    Parameters
    ----------
    topology : str
        Path to a topology file (e.g., PSF, TPR, LAMMPSDATA).
    trajectory : str or None
        Path to a trajectory file (e.g., DCD, XTC, TRR)

    Returns
    -------
    dict
        Dictionary with validation results containing the following keys:

        - ``valid_topology`` : bool
          Topology could be loaded.
        - ``need_bonds`` : bool
          True if bonds are required (residue size >= 2).
        - ``need_angles`` : bool
          True if angles are required (residue size >= 3).
        - ``has_bonds`` : bool
          True if bonds are present in the topology.
        - ``has_angles`` : bool
          True if angles are present in the topology.
        - ``valid_trajectory`` : bool
          Trajectory could be loaded.
        - ``match`` : bool
          Number of atoms in trajectory matches topology.

    Notes
    -----
    Prints warnings and error messages if requirements are not met.

    """
    out = {
        "valid_topology": False,
        "need_bonds": False,
        "need_angles": False,
        "has_bonds": False,
        "has_angles": False,
        "valid_trajectory": False,
        "match": False,
    }

    if not topology:
        print("Please provide a Topology File")
        return out

    # Load topology only
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"No coordinate reader found for .*",
                category=UserWarning,
            )
            u_top = mda.Universe(topology=topology)
        out["valid_topology"] = True
    except FileNotFoundError as e:
        print("Topology not found")
        print(e)
        return out

    # --- Decide requirements based on residue sizes
    max_atoms_in_residue = max((len(res.atoms) for res in u_top.residues), default=0)
    out["need_bonds"] = max_atoms_in_residue >= 2
    out["need_angles"] = max_atoms_in_residue >= 3

    # --- Check what's present
    if out["need_bonds"]:
        try:
            out["has_bonds"] = len(u_top.bonds) > 0
        except NoDataError as e:
            print("No Bonds detected in Topology with Residues with multiple Atoms")
            print(e)
    if out["need_angles"]:
        try:
            out["has_angles"] = len(u_top.angles) > 0
        except NoDataError as e:
            print("No Angles detected in Topology with Residues with 3 or more Atoms")
            print(e)
    # --- Trajectory check
    if trajectory:
        try:
            n_top = u_top.atoms.n_atoms
            try:
                u_top.load_new(trajectory)  # reuse same Universe
            except TypeError:
                u_top = mda.Universe(topology, trajectory, format="LAMMPSDUMP")
            out["valid_trajectory"] = True
            out["match"] = u_top.atoms.n_atoms == n_top
        except (OSError, ValueError) as e:
            print("The Trajectory does not match the Topology")
            print(e)

    else:
        print("Please provide a Trajectory File")
    return out


def _requirements_met(out):
    ok = out["valid_topology"]
    if out["need_bonds"]:
        ok = ok and out["has_bonds"]
    if out["need_angles"]:
        ok = ok and out["has_angles"]
    ok = ok and out["valid_trajectory"] and out["match"]
    return ok
