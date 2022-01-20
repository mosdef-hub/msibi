HOOMD2_HEADER = """
import hoomd
import hoomd.md
from hoomd.init import read_gsd

hoomd.context.initialize("")
system = read_gsd("{0}", frame=-1, time_step=0)

pot_width = {1:d}
nl = hoomd.md.nlist.cell()
nl.reset_exclusions(exclusions=["1-2", "1-3", "1-4"])
table = hoomd.md.pair.table(width=pot_width, nlist=nl)
"""

HOOMD_TABLE_ENTRY = """
table.set_from_file('{type1}', '{type2}', filename='{potential_file}')
"""

HOOMD_TEMPLATE = """
_all = hoomd.group.all()
hoomd.md.integrate.mode_standard({dt})
integrator_kwargs = {integrator_kwargs}
integrator = {integrator}(group=_all, **integrator_kwargs)


hoomd.dump.gsd(
    filename="query.gsd",
    group=_all,
    period={gsd_period},
    overwrite=True,
    dynamic=["momentum"]
    )

hoomd.run({n_steps})
"""
