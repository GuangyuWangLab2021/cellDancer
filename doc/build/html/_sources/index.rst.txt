cellDancer - High resolution of RNA velocity
===========================================================================================
**cellDancer** is a tool for RNA velocity analysis in single cells.

RNA velocity has provided a new way to construct directed dynamics of cellular states by modeling dynamics of pre-mRNAs (unspliced) and mature mRNAs (spliced) from transcriptional dynamic kinetics (`La Manno et al., Nature, 2018 <https://doi.org/10.1038/s41586-018-0414-6>`_).

Here, we developed cellDancer, a deep learning framework, to decouple each cell’s transcription, splicing, and degradation rates rather than jointly estimating a set of shared rates for all cells.

cellDancer key applications
============================
* Estimate RNA velocity for each gene.
* Derive cell fates in embedding space.
* Estimate pseudotime for each cell in embedding space.

Latest news
^^^^^^^^^^^^^^^^^^^^
- to be updated.

.. toctree::
    :caption: cellDancer
    :titlesonly:

    about
    API

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/install
   notebooks/Data Preparation
   notebooks/CaseStudy_Gastrulation_transcriptional_boost
   notebooks/Neuro_velocity_analysis
   notebooks/Pancreas_decoding_kinetics
