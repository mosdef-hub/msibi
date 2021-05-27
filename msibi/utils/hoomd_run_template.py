all = hoomd.group.all()
nvt_int = hoomd.md.integrate.langevin(group=all, kT=T_final, seed=1)
hoomd.md.integrate.mode_standard(dt=0.001)

hoomd.dump.gsd(
    filename="query.gsd",
    group=all,
    period=1000,
    overwrite=True)
hoomd.run(1e5)
