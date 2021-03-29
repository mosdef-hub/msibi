""""MSIBI: A package for optimizing coarse-grained force fields using multistate
iterative Boltzmann inversion.

"""

import os
from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys

try:
    import mdtraj
except ImportError:
    print(
        "Building and running msibi requires mdtraj. See "
        "http://mdtraj.org/latest/installation.html for help!"
    )
    sys.exit(1)

requirements = ["numpy", "six", "networkx"]

NAME = "msibi"
# Load the package's __version__.py module as a dictionary.
here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        errcode = pytest.main(["msibi"])
        sys.exit(errcode)


setup(
    name=NAME,
    version=about["__version__"],
    description=(
        "A package for optimizing coarse-grained force fields using " +
        "multistate iterative Boltzmann inversion."
        ),
    url="http://github.com/cmelab/msibi",
    author="Christoph Klein, Timothy C. Moore",
    author_email=(
        "christoph.klein@vanderbilt.edu, timothy.c.moore@vanderbilt.edu"
        ),
    license="MIT",
    packages=["msibi"],
    install_requires=requirements,
    zip_safe=False,
    test_suite="tests",
    cmdclass={"test": PyTest},
    extras_require={"utils": ["pytest"]},
)
