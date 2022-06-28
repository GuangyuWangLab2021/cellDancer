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

Installation
========================================================
cellDancer requires Python version >= 3.7.6 to run.

To run cellDancer locally, create an [conda](https://docs.conda.io/en/latest/) or [Anaconda](https://www.anaconda.com/) environment as ```conda create -n cellDancer python==3.7.6```, and activate the new environment with ```conda activate cellDancer```. Then install the dependencies by ```pip install -r requirements.txt``` using [requirememts.txt](requirememts.txt).

To install cellDancer from source code, run:
``pip install your_path/cellDancer``
