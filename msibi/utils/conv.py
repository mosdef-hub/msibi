import MDAnalysis as mda
import gsd.hoomd
import numpy as np

def _mda_check(topology, trajectory):
    """
    MDAnalysis input check:
      - Topology must load.
      - Bonds required if any residue has >=2 atoms.
      - Angles required if any residue has >=3 atoms.
      - If trajectory given, it must load with the topology and atom counts must match.
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
        return out

    # Load topology only
    try:
        u_top = mda.Universe(topology)
        out["valid_topology"] = True
    except:
        return out

    # --- Decide requirements based on residue sizes
    max_atoms_in_residue = max((len(res.atoms) for res in u_top.residues), default=0)
    out["need_bonds"] = max_atoms_in_residue >= 2
    out["need_angles"] = max_atoms_in_residue >= 3

    # --- Check what's present
    out["has_bonds"] = len(u_top.bonds) > 0
    out["has_angles"] = len(u_top.angles) > 0

    # --- Trajectory check
    if trajectory:
        try:
            u = mda.Universe(topology, trajectory)
            out["valid_trajectory"] = True
            out["match"] = (u.atoms.n_atoms == u_top.atoms.n_atoms)
        except:
            pass

    return out


def _requirements_met(out, require_traj=False):
    ok = out["valid_topology"]
    if out["need_bonds"]:
        ok = ok and out["has_bonds"]
    if out["need_angles"]:
        ok = ok and out["has_angles"]
    if require_traj:
        ok = ok and out["valid_trajectory"] and out["match"]
    return ok


def gsd_from_files(topology_file, traj_file, output='output.gsd'):
    """
    Converts an MDAnalysis.Universe if checks pass; otherwise prints unmet
    requirements.
    """
    out = _mda_check(topology_file, traj_file)
    require_traj = traj_file is not None
    ok = _requirements_met(out, require_traj=require_traj)

    if ok:
        gsd_from_universe(mda.Universe(topology_file, traj_file, output))

    if not ok:
        print("Result:", out)

    
def gsd_from_universe(universe, output='output.gsd'):
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

