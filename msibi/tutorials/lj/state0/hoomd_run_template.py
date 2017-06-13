# **** 2017_05_31 Notes ******
# update file to correct hoomd 2.0 syntax
# look up docs
# combine with state.py in main folder


all = hoomd.group.all()
nvt_int = hoomd.md.integrate.langevin(group=all, kT=T_final, seed=1) # possibly use kT instead of T???
hoomd.md.integrate.mode_standard(dt=0.001) #integrate.\*_rigid() no longer exists. Use a standard integrator on group.rigid_center(), and define rigid bodies using constrain.rigid()


hoomd.run(1e2)
output_dcd = hoomd.dump.dcd(filename='query.dcd', period=100, overwrite=True)
hoomd.run(1e4)
