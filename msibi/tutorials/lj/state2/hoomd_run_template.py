all = hoomd.group.all()
nvt_int = hoomd.md.integrate.langevin(group=all, kT=T_final, seed=1)
hoomd.md.integrate.mode_standard(dt=0.001)

hoomd.run(1e2)
output_dcd = hoomd.dump.dcd(filename='query.dcd', period=100, overwrite=True)
hoomd.run(1e4)
