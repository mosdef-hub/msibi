import os

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from cmeutils.structure import gsd_rdf

from msibi.potentials import alpha_array, head_correction, tail_correction
from msibi.utils.error_calculation import calc_similarity
from msibi.utils.exceptions import UnsupportedEngine
from msibi.utils.find_exclusions import find_1_n_exclusions
from msibi.utils.general import find_nearest
from msibi.utils.smoothing import savitzky_golay


class Pair(object):
    """A pair interaction to be optimized.

    Parameters
    ----------
    type1 : str, required
        The name of one particle type on the particle pair.
        Must match the names found in the State's .gsd trajectory file.
        See  gsd.hoomd.ParticleData.types
    type2 : str, required
        The name of one particle type on the particle pair.
        Must match the names found in the State's .gsd trajectory file.
        See  gsd.hoomd.ParticleData.types
    potential :


    Attributes
    ----------
    name : str
        Pair name.
    potential : func
        Values of the potential at every pot_r.
    """

    def __init__(self, type1, type2, potential, head_correction_form="linear"):
        self.type1 = str(type1)
        self.type2 = str(type2)
        self.name = f"{self.type1}-{self.type2}"
        self.potential_file = ""
        self._states = dict()
        if isinstance(potential, str):
            self.potential = np.loadtxt(potential)[:, 1]
            # TODO: this could be dangerous
        else:
            self.potential = potential
        self.previous_potential = None
        self.head_correction_form = head_correction_form

    def _add_state(self, state, smooth=True):
        """Add a state to be used in optimizing this pair.

        Parameters
        ----------
        state : msibi.state.State
            A state object created previously.
        """
        target_rdf = self.get_state_rdf(state, query=False)
        if state._opt.smooth_rdfs:
            target_rdf[:, 1] = savitzky_golay(
                target_rdf[:, 1], 9, 2, deriv=0, rate=1
                )
            negative_idx = np.where(target_rdf < 0)
            target_rdf[negative_idx] = 0
            
        self._states[state] = {
            "target_rdf": target_rdf,
            "current_rdf": None,
            "alpha": state.alpha,
            "alpha_form": "linear",
            "pair_indices": None,
            "f_fit": [],
            "path": state.dir
        }

    def get_state_rdf(self, state, query):
        """Calculate the RDF of a Pair at a State."""
        if query:
            traj = state.query_traj
        else:
            traj = state.traj_file
        rdf, norm = gsd_rdf(
                traj,
                self.type1,
                self.type2,
                start=-state._opt.max_frames,
                r_max=state._opt.rdf_cutoff,
                bins=state._opt.n_rdf_points,
                exclude_bonded=state._opt.rdf_exclude_bonded
                )
        return np.stack((rdf.bin_centers, rdf.rdf*norm)).T

    def compute_current_rdf(
        self,
        state,
        smooth,
        verbose=False
        ):

        rdf = self.get_state_rdf(state, query=True)
        self._states[state]["current_rdf"] = rdf

        if state._opt.smooth_rdfs:
            current_rdf = self._states[state]["current_rdf"]
            current_rdf[:, 1] = savitzky_golay(
                current_rdf[:, 1], 9, 2, deriv=0, rate=1
            )
            negative_idx = np.where(current_rdf < 0)
            current_rdf[negative_idx] = 0
            if verbose:  # pragma: no cover
                plt.title(f"RDF smoothing for {state.name}")
                plt.plot(rdf[:,0], rdf[:, 1], label="unsmoothed")
                plt.plot(rdf[:,0], current_rdf[:,1], label="smoothed")
                plt.legend()
                plt.show()

        # Compute fitness function comparing the two RDFs.
        f_fit = calc_similarity(
            rdf[:, 1], self._states[state]["target_rdf"][:, 1]
        )
        self._states[state]["f_fit"].append(f_fit)


    def save_current_rdf(self, state, iteration, dr):
        """Save the current rdf

        Parameters
        ----------
        state : State
            A state object
        iteration : int
            Current iteration step, used in the filename
        dr : float
            The RDF bin size
        """
        rdf = self._states[state]["current_rdf"]
        rdf[:, 0] -= dr / 2
        np.savetxt(os.path.join(
            state.dir,
            f"pair_{self.name}-state_{state.name}-step{iteration}.txt"
            ),
            rdf)

    def update_potential(self, pot_r, r_switch=None, verbose=False):
        """Update the potential using all states. """
        self.previous_potential = np.copy(self.potential)
        for state in self._states:
            kT = state.kT
            alpha0 = self._states[state]["alpha"]
            form = self._states[state]["alpha_form"]
            alpha = alpha_array(alpha0, pot_r, form=form)

            current_rdf = self._states[state]["current_rdf"]
            target_rdf = self._states[state]["target_rdf"]

            # For cases where rdf_cutoff != pot_cutoff, only update the
            # potential using RDF values < pot_cutoff.
            unused_rdf_vals = current_rdf.shape[0] - self.potential.shape[0]
            if unused_rdf_vals != 0:
                current_rdf = current_rdf[:-unused_rdf_vals,:]
                target_rdf = target_rdf[:-unused_rdf_vals,:]

            if verbose:  # pragma: no cover
                plt.plot(current_rdf[:,0], current_rdf[:,1], label="current rdf")
                plt.plot(target_rdf[:,0], target_rdf[:,1], label="target rdf")
                plt.legend()
                plt.show()

            # The actual IBI step.
            self.potential += (
                    kT * alpha * np.log(current_rdf[:,1] / target_rdf[:,1]) / len(self._states)
            )

            if verbose:  # pragma: no cover
                plt.plot(
                    pot_r, self.previous_potential, label="previous potential"
                )
                plt.plot(pot_r, self.potential, label="potential")
                plt.ylim(
                    (min(self.potential[np.isfinite(self.potential)])-1,10)
                )
                plt.legend()
                plt.show()

        # Apply corrections to ensure continuous, well-behaved potentials.
        pot = self.potential
        self.potential = tail_correction(pot_r, self.potential, r_switch)
        tail = self.potential
        self.potential = head_correction(
            pot_r,
            self.potential,
            self.previous_potential,
            self.head_correction_form
        )
        head = self.potential
        if verbose:  # pragma: no cover
            plt.plot(pot_r, head, label="head correction")
            plt.plot(pot_r, pot, label="uncorrected potential")
            idx_r, _ = find_nearest(pot_r, r_switch)
            plt.plot(pot_r[idx_r:], tail[idx_r:], label="tail correction")

            plt.ylim((min(pot[np.isfinite(pot)])-1, 10))
            plt.legend()
            plt.show()

    def save_table_potential(self, r, dr, iteration=0, engine="hoomd"):
        """Save the table potential to a file usable by the MD engine. """
        V = self.potential
        F = -1.0 * np.gradient(V, dr)
        data = np.vstack([r, V, F])

        basename = os.path.basename(self.potential_file)
        basename = "step{0:d}.{1}".format(iteration, basename)
        dirname = os.path.dirname(self.potential_file)
        iteration_filename = os.path.join(dirname, basename)

        # This file is overwritten at each iteration and actually used for
        # performing the query simulations.
        np.savetxt(self.potential_file, data.T)
        # This file is written for viewing of how the potential evolves.
        np.savetxt(iteration_filename, data.T)
