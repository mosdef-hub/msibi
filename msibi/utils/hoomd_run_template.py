HOOMD2_HEADER = """
import hoomd
import hoomd.md
from hoomd.init import read_gsd

hoomd.context.initialize("")
system = read_gsd("{0}", frame=-1, time_step=0)

pot_width = {1:d}
nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=pot_width, nlist=nl)
"""

HOOMD_TABLE_ENTRY = """
table.set_from_file('{type1}', '{type2}', filename='{potential_file}')
"""

HOOMD_BOND_INIT = """
harmonic_bond = hoomd.md.bond.harmonic()
"""

HOOMD_BOND_ENTRY = """
harmonic_bond.bond_coeff.set('{name}', k={k}, r0={r0})
"""

HOOMD_ANGLE_INIT = """
harmonic_angle = hoomd.md.angle.harmonic()
"""

HOOMD_ANGLE_ENTRY = """
harmonic_angle.angle_coeff.set('{name}', k={k}, t0={theta})
"""

HOOMD_TEMPLATE = """
_all = hoomd.group.all()
hoomd.md.integrate.mode_standard({dt})
integrator = {_integrator}

hoomd.dump.gsd(
    filename="query.gsd",
    group=_all,
    period={gsd_period},
    overwrite=True,
    dynamic=["momentum"]
    )

hoomd.run({n_steps})
"""
