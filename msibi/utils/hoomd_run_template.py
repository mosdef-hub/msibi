HOOMD2_HEADER = """
import hoomd
import hoomd.md
from hoomd.init import read_gsd

hoomd.context.initialize("")
system = read_gsd("{0}", frame=-1, time_step=0)

nl = {1}()
nl.reset_exclusions(exclusions={2})
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
