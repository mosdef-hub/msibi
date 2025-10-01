Tutorials
=========

Here are some tutorials which go further in depth into MSIBI use cases. For quick-start examples of code using MSIBI's API see :doc:`examples`.

The introduction tutorial uses `.gsd` `files <https://github.com/mosdef-hub/msibi/tree/main/msibi/tests/validation/target_data>`_ that are stored in the source repository. These are not packaged with the
conda-forge distribution of `msibi`, and depending on how you installed `msibi` from source, they may or may not be accessible. To ensure these files are accessible when running this tutorial, please use the following installation steps.

This uses the `anaconda package manager <https://www.anaconda.com/download>`_.

.. code-block:: bash

   git clone https://github.com/mosdef-hub/msibi
   cd msibi
   conda env create -f environment.yml python=3.13 jupyter
   conda activate msibi
   pip install -e .

.. toctree::
    :maxdepth: 1

    tutorials/01-Introduction
