from cmeutils.structure import gsd_rdf
import numpy as np

def state_pair_target_rdf(state, pair, exclude_bonded=False):
    """ Calculate and store the RDF data from a trajectory file of a particular State. """ 
    rdf, norm = gsd_rdf(
            state.traj_file,
            pair.type1,
            pair.type2,
            start=-5,
            r_max=4,
            bins=100,
            exclude_bonded=exclude_bonded
            )
    return np.stack((rdf.bin_centers, rdf.rdf*norm)).T

