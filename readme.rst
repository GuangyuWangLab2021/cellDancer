cellDancer - Estimating Cell-dependent RNA Velocity
===========================================================================================

**cellDancer** is a modularized, parallelized, and scalable tool based on a deep learning framework for the RNA velocity analysis of scRNA-seq.

.. image:: _static/training_progress.png
  :width: 100%
  :alt: cell_type_u_s_sample_df

cellDancer's key applications
========================================================
* Estimate RNA velocity for each gene.
* Derive cell fates in embedding space.
* Estimate pseudotime for each cell in embedding space.

What's new
========================================================
cellDancer is updated to v1.1.3

* Added API ``celldancer.utilities.to_dynamo`` and ``elldancer.utilities.map_cd_velocity_to_dynamo`` to import cellDancer results to Dynamo.
* Added deep learning parameters n_neighbors, dt, and learning_rate in function ``cellDancer.velocity()``.
* Added new loss function: mix, rmse in function ``cellDancer.velocity()``.

Installation
========================================================
cellDancer requires Python version >= 3.7.6 to run.

To run cellDancer locally, create an `conda <https://docs.conda.io/en/latest>`_ or `Anaconda <https://www.anaconda.com/>`_ environment as ``conda create -n cellDancer python==3.7.6``, and activate the new environment with ``conda activate cellDancer``. Then install the dependencies by ``pip install -r requirements.txt`` using `requirememts.txt <https://drive.google.com/file/d/1-yhX3yioYOJEsuYimnb8ja9AP4OKjczC/view>`_ .

To install cellDancer from source code, run:
``pip install 'your_path/Source Code/cellDancer'``
