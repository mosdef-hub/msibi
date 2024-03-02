<img src="/docs/msibi.png" height="400">

# MultiState Iterative Boltzmann Inversion (MS-IBI)
----------------------------------------
[![pytest](https://github.com/cmelab/msibi/actions/workflows/pytest.yml/badge.svg)](https://github.com/cmelab/msibi/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/cmelab/msibi/branch/master/graph/badge.svg?token=7NFPBMBN0I)](https://codecov.io/gh/cmelab/msibi)

A package to help you manage and run pair potential optimizations using multistate iterative Boltzmann inversion.


### Install from source:
```bash
git clone https://github.com/cmelab/msibi.git
cd msibi
conda env create -f environment.yml
conda activate msibi
pip install .
```


### Citation [![Citing MSIBI](https://img.shields.io/badge/DOI-10.1063%2F1.4880555-blue.svg)](http://dx.doi.org/10.1063/1.4880555)
Details of the underlying method and its validation can be found [here](http://dx.doi.org/10.1063/1.4880555).

If you use this package, please cite the above paper. The BibTeX reference is
```
@article{Moore2014,
      author = "Moore, Timothy C. and Iacovella, Christopher R. and McCabe, Clare",
      title = "Derivation of coarse-grained potentials via multistate iterative Boltzmann inversion",
      journal = "The Journal of Chemical Physics",
      year = "2014",
      volume = "140",
      number = "22", 
      doi = "http://dx.doi.org/10.1063/1.4880555" 
}
```

