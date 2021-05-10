from cmeutils.structure import gsd_rdf
import numpy as np

def state_pair_rdf(state, pair, exclude_bonded=True):
    """ Calculate and store the RDF data from a trajectory file of a particular State. """ 
    rdf, norm = gsd_rdf(
            state.traj_file,
            pair.type1,
            pair.type2,
            start=-state.opt.max_frames,
            r_max=state.opt.rdf_cutoff,
            bins=state.opt.n_rdf_points,
            exclude_bonded=exclude_bonded
            )
    return np.stack((rdf.bin_centers, rdf.rdf*norm)).T

