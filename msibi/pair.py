import os

import matplotlib.pyplot as plt
import numpy as np
from cmeutils.structure import gsd_rdf

from msibi.potentials import alpha_array, head_correction, tail_correction
from msibi.utils.error_calculation import calc_similarity
from msibi.utils.exceptions import UnsupportedEngine
from msibi.utils.general import find_nearest
from msibi.utils.smoothing import savitzky_golay


LJ_PAIR_ENTRY = "lj.pair_coeff.set('{}', '{}', epsilon={}, sigma={}, r_cut={})"
MORSE_PAIR_ENTRY = "morse.pair_coeff.set('{}','{}',D0={},alpha={},r0={},r_cut={}"
GAUSS_PAIR_ENTRY = "gauss.pair_coeff.set('{}', '{}', epsilon={}, sigma={}, r_cut={})"
TABLE_PAIR_ENTRY = "table.set_from_file('{}', '{}', filename='{}')"


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

    Attributes
    ----------
    name : str
        Pair name.
    potential : func
        Values of the potential at every r_range.

    """

    def __init__(self, type1, type2, head_correction_form="linear"):
        self.type1 = str(type1)
        self.type2 = str(type2)
        self.name = f"{self.type1}-{self.type2}"
        self.potential_file = ""
        self._states = dict()
        self.previous_potential = None
        self.head_correction_form = head_correction_form

    def set_lj(self, epsilon, sigma, r_cut):
        """Creates a hoomd 12-6 LJ pair potential used during
        the query simulations. This method is not compatible when
        optimizing pair potentials. Rather, this method should
        only be used to create static pair potentials while optimizing
        Bonds or Angles.

        Parameters
        ----------
        epsilon : float, required
            Sets the dept hof the potential energy well.
        sigma : float, required
            Sets the particle size.
        r_cut : float, required
            Maximum distance used to calculate neighbor pair potentials.

        """
        self.pair_type = "hoomd_lj"
        self.pair_init = "lj = pair.lj(nlist=nl)"
        self.pair_entry = LJ_PAIR_ENTRY.format(
                self.type1, self.type2, epsilon, sigma, r_cut
        )

    def set_morse(self, D0, alpha, r0, r_cut):
        """Creates a hoomd Morse pair potential used during
        the query simulations. This method is not compatible when
        optimizing pair potentials. Rather, this method should
        only be used to create static pair potentials while optimizing
        Bonds or Angles.

        Parameters
        ----------
        D0 : float, required
            The depth of the potential well at it's minimum point.
        alpha : float, required
            Sets the width of the potential well.
        r0 : float, required
            The position of the potential minimum.
        r_cut : float, required
            Maximum distance used to calculate neighbor pair potentials.

        """
        self.pair_type = "hoomd_morse"
        self.pair_init = f"morse = pair.morse(nlist=nl)"
        self.pair_entry = MORSE_PAIR_ENTRY.format(
                self.type1, self.type2, D0, alpha, r0, r_cut
        )

    def set_gauss(self, epsilon, sigma, r_cut):
        """Creates a hoomd Gaussian pair potential used during
        the query simulations. This method is not compatible when
        optimizing pair potentials. Rather, this method should
        only be used to create static pair potentials while optimizing
        Bonds or Angles.

        Parameters
        ----------
        epsilon : float, required
            Sets the dept hof the potential energy well.
        sigma : float, required
            Sets the particle size.
        r_cut : float, required
            Maximum distance used to calculate neighbor pair potentials.

        """
        self.pair_type = "hoomd_gauss"
        self.pair_init = f"gauss = pair.gauss(nlist=nl)"
        self.pair_entry = GAUSS_PAIR_ENTRY.format(
                self.type1, self.type2, epsilon, sigma, r_cut
        )

    def set_table_potential(
            self, epsilon, sigma, r_min, r_max, n_points, m=12, n=6
    ):
        """Creates a table potential V(r) over the range r_min - r_max.

        Uses the Morse potential functional form.

        This should be the pair potential form of choice when optimizing
        pairs; however, you can also use this method to set a static
        pair potential while optimizing other potentials such as
        Angles and Bonds.

        Parameters
        ----------
        epsilon : float, required
            Sets the dept hof the potential energy well.
        sigma : float, required
            Sets the particle size.
        r_min : float, required
            Sets the lower bound for distances used in the table potential
        r_max : float, required
            Sets the upper bound for distances used in the table potential
        n_points : int, required
            Sets the number of points between r_min and r_max to extrapolate
            the table potential
        m : int, default = 12
            The exponent of the repulsive term
        n : int, default = 6
            The exponent of the attractive term

        """
        def create_pair_table(r, eps, sig, m, n):
            """The Morse potential functional form"""
            prefactor = (m / (m - n)) * (m / n) ** (n / (m - n))
            return prefactor * eps * ((sig / r) ** m - (sig / r) ** n)

        self.r_min = r_min
        self.r_max = r_max
        self.n_points = int(n_points)
        self.dr = (r_max - r_min) / self.n_points
        self.r_range = np.arange(r_min, r_max + self.dr, self.dr)
        self.potential = create_pair_table(self.r_range, epsilon, sigma, m, n)
        self.pair_type = "table"
        self.pair_init = f"table=hoomd.md.pair.table(width={self.n_points},nlist=nl)"
        self.pair_entry = TABLE_PAIR_ENTRY.format(
                self.type1, self.type2, self.potential_file
        )

    def set_from_file(self, file_path):
        # TODO: Finish support for loading pair pot from file
        self.potential_file = file_path
        self.pair_type = "file"
        self.pair_init = ""
        self.pair_entry = ""

    def _add_state(self, state, smooth=True):
        """Add a state to be used in optimizing this pair.

        Parameters
        ----------
        state : msibi.state.State
            A state object created previously.
        """
        if state._opt.optimization == "pairs":
            target_rdf = self._get_state_rdf(state, query=False)
            if state._opt.smooth_rdfs:
                target_rdf[:, 1] = savitzky_golay(
                    target_rdf[:, 1], 9, 2, deriv=0, rate=1
                )
                negative_idx = np.where(target_rdf < 0)
                target_rdf[negative_idx] = 0
        else:
            target_rdf = None

        self._states[state] = {
            "target_rdf": target_rdf,
            "current_rdf": None,
            "alpha": state.alpha,
            "alpha_form": "linear",
            "f_fit": [],
            "path": state.dir
        }

    def _get_state_rdf(self, state, query):
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
            r_max=self.r_max,
            bins=self.n_points,
            exclude_bonded=state._opt.rdf_exclude_bonded
        )
        return np.stack((rdf.bin_centers, rdf.rdf*norm)).T

    def _compute_current_rdf(self, state, smooth, verbose=False):
        """Calcualte the current RDF from the query trajectory.
        Updates the 'current_rdf' value in this Pair's state dict.
        Applies smoothing if applicable and calculates the f_fit between
        the current RDF and target RDF.

        """
        rdf = self._get_state_rdf(state, query=True)
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

    def _save_current_rdf(self, state, iteration):
        """Save the current rdf

        Parameters
        ----------
        state : State
            A state object
        iteration : int
            Current iteration step, used in the filename

        """
        rdf = self._states[state]["current_rdf"]
        rdf[:, 0] -= self.dr / 2

        fname = f"pair_pot_{self.name}-state_{state.name}-step{iteration}.txt"
        fpath = os.path.join(state.dir, fname)
        np.savetxt(fpath, rdf)

    def _update_potential(self, verbose=False):
        """Update the potential using all states. """
        self.previous_potential = np.copy(self.potential)
        for state in self._states:
            kT = state.kT
            alpha0 = self._states[state]["alpha"]
            form = self._states[state]["alpha_form"]
            alpha = alpha_array(alpha0, self.r_range, form=form)
            N = len(self._states)
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
                    kT * alpha * np.log(current_rdf[:,1] / target_rdf[:,1]) / N 
            )

            if verbose:  # pragma: no cover
                plt.plot(
                    self.r_range, self.previous_potential, label="previous potential"
                )
                plt.plot(self.r_range, self.potential, label="potential")
                plt.ylim(
                    (min(self.potential[np.isfinite(self.potential)])-1,10)
                )
                plt.legend()
                plt.show()

        # Apply corrections to ensure continuous, well-behaved potentials.
        pot = self.potential
        self.potential = tail_correction(self.r_range, self.potential, self.r_switch)
        tail = self.potential
        self.potential = head_correction(
            self.r_range,
            self.potential,
            self.previous_potential,
            self.head_correction_form
        )
        head = self.potential
        if verbose:  # pragma: no cover
            plt.plot(self.r_range, head, label="head correction")
            plt.plot(self.r_range, pot, label="uncorrected potential")
            idx_r, _ = find_nearest(self.r_range, self.r_switch)
            plt.plot(
                    self.r_range[idx_r:], tail[idx_r:], label="tail correction"
            )

            plt.ylim((min(pot[np.isfinite(pot)])-1, 10))
            plt.legend()
            plt.show()

    def _save_table_potential(self, r, dr, iteration=0):
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
