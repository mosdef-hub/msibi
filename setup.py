""""MSIBI: A package for optimizing coarse-grained force fields using multistate
iterative Boltzmann inversion.

"""

import os
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys


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
    author="Chris Jones",
    author_email=("chrisjones4@u.boisestate.edu"),
    packages=find_packages(
        exclude=("tests", "docs")
    ),
    package_data={
        "msibi":[
            "msibi/**"
        ]
    },
    license="MIT",
    zip_safe=False,
    test_suite="tests",
    cmdclass={"test": PyTest},
    extras_require={"utils": ["pytest"]},
)
