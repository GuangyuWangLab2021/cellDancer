cellDancer - Estimating Cell-dependent RNA Velocity
===========================================================================================

**cellDancer** is a modularized, parallelized, and scalable tool based on a deep learning framework for the RNA velocity analysis of scRNA-seq. Our website of tutorials is available at `cellDancer Website <https://guangyuwanglab2021.github.io/cellDancer_website/>`_.


cellDancer's key applications
========================================================
* Estimate cell-specific RNA velocity for each gene.
* Derive cell fates in embedding space.
* Estimate pseudotime for each cell in embedding space.

What's new
========================================================
cellDancer is updated to v1.1.4

* Released cellDancer at PyPI. Mainly updated requirements.txt and setup.py.

cellDancer is updated to v1.1.3

* Added ``celldancer.utilities.to_dynamo`` and ``celldancer.utilities.export_velocity_to_dynamo`` to import cellDancer results to dynamo.
* Added deep learning parameters n_neighbors, dt, and learning_rate in function ``cellDancer.velocity()``.
* Added new loss function: mix, rmse in function ``cellDancer.velocity()``.

Installation
========================================================
cellDancer requires Python version >= 3.7.6 to run.

To run cellDancer locally, create an `conda <https://docs.conda.io/en/latest>`_ or `Anaconda <https://www.anaconda.com/>`_ environment as ``conda create -n cellDancer python==3.7.6``, and activate the new environment with ``conda activate cellDancer``. cellDancer could be installed with ``pip install celldancer``.

To install cellDancer from source code, run:
``pip install 'your_path/Source Code/cellDancer'``

The dependencies could also be installed by ``pip install -r requirements.txt``.
