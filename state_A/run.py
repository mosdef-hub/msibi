
import hoomd
import hoomd.md
from hoomd.deprecated.init import read_xml

hoomd.context.initialize("")
system = read_xml(filename="start.hoomdxml", wrap_coordinates=True)
T_final = 0.5

pot_width = 121
nl = hoomd.md.nlist.cell()
table = hoomd.md.pair.table(width=pot_width, nlist=nl)


table.set_from_file('C3', 'C3', filename='/Users/davyyue/Research/msibi/msibi/tutorials/propane/potentials/pot.C3-C3.txt')
all = hoomd.group.all()
nvt_int = hoomd.md.integrate.langevin(group=all, kT=T_final, seed=1)
hoomd.md.integrate.mode_standard(dt=0.001)

hoomd.run(1e2)
output_dcd = hoomd.dump.dcd(filename='query.dcd', period=100, overwrite=True)
hoomd.run(1e4)
