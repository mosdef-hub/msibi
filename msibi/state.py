import mdtraj as md


class State(object):
    """A single state used as part of a multistate optimization.

    Attributes
    ----------
    k : float
        Boltzmann's  constant in specified units.
    T : float
        Temperature in kelvin.
    traj : md.Trajectory
        The trajectory associated with this state.

    """
    def __init__(self, k, T, traj_path, top_path=None):
        self.traj_path = traj_path
        self.top_path = top_path
        self.kT = k * T

    def reload_query_trajectory(self):
        if self.top_path:
            self.traj = md.load(self.traj_path, topology=top_path)
        else:
            self.traj = md.load(traj_path)


