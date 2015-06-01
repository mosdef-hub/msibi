all = group.all()
nvt_int = integrate.bdnvt(group=all, T=T_final)
integrate.mode_standard(dt=0.001)


run(1e2)
output_dcd = dump.dcd(filename='query.dcd', period=100, overwrite=True)
run(1e3)

output_xml = dump.xml()
output_xml.set_params(all=True)
output_xml.write(filename='final.xml')
output_pdb = dump.pdb()
output_pdb.write(filename='final.pdb')
