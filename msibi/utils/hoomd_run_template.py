HOOMD3_HEADER = """
import hoomd

device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device)
sim.create_state_from_snapshot({0})
nl = {1}()
integrator = hoomd.md.Integrator(dt={2})
method_kwargs = {3}
method = {method}(
    kT={4},
    filter=hoomd.filter.All(),
    **method_kwargs
)
integrator.methods.append(method)

sim.operations.integrator = integrator
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT={4})
"""

HOOMD3_TEMPLATE = """

"""

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
