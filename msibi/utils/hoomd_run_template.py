all = hoomd.group.all()
nvt_int = hoomd.md.integrate.langevin(group=all, kT=T_final, seed=1)
hoomd.md.integrate.mode_standard(dt=0.001)

hoomd.run(1e2)
output_gsd = hoomd.dump.gsd(filename="query.gsd", period=100, overwrite=True)
hoomd.run(1e4)
