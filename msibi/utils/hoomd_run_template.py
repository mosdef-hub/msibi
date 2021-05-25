all = hoomd.group.all()
nvt_int = hoomd.md.integrate.langevin(group=all, kT=T_final, seed=1)
hoomd.md.integrate.mode_standard(dt=0.001)

hoomd.run(1e4)
hoomd.dump.gsd(
    filename="query.gsd",
    group=all,
    period=100,
    overwrite=True)
hoomd.run(2e5)
