from hoomd_script import *

system = init.read_xml(filename="start.xml")
T_final = 1.5

pot_width = 41
table = pair.table(width=pot_width)

table.set_from_file('1', '1', filename='/Users/tcmoore3/programs/msibi/msibi/tests/lj/potentials/pot.1-1.txt')
all = group.all()
nvt_int = integrate.bdnvt(group=all, T=T_final)
integrate.mode_standard(dt=0.001)


run(1e3)
output_dcd = dump.dcd(filename='query.dcd', period=100, overwrite=True)
run(1e3)

output_xml = dump.xml()
output_xml.set_params(all=True)
output_xml.write(filename='final.xml')
output_pdb = dump.pdb()
output_pdb.write(filename='final.pdb')
