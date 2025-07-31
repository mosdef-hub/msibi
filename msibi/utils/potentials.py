def polynomial_potential(x, x0, k4, k3, k2):
    """Creates a polynomial potential with the following form

        V(x) = k4(x-x0)^4 + k3(x-x0)^3 + k2(x-x0)^2

    Can be used in creating table potentials.

    """
    V_x = k4 * ((x - x0) ** 4) + k3 * ((x - x0) ** 3) + k2 * ((x - x0) ** 2)
    return V_x


def mie(r, epsilon, sigma, m, n):
    """The Mie potential functional form.

    Can be used for creating table Mie potentials.
    """
    prefactor = (m / (m - n)) * (m / n) ** (n / (m - n))
    V_r = prefactor * epsilon * ((sigma / r) ** m - (sigma / r) ** n)
    return V_r


def lennard_jones(r, epsilon, sigma):
    """Create an LJ 12-6 table potential."""
    return mie(r=r, epsilon=epsilon, sigma=sigma, m=12, n=6)


def alpha_array(alpha0, pot_r, dr, form="linear"):
    """Generate an array of alpha values used for scaling in the IBI step."""
    return alpha0 * (1.0 - (pot_r - dr) / (pot_r[-1] - dr))
