.. toolkit documentation master file, created by
   sphinx-quickstart on Wed Feb  9 17:10:01 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

About cellDancer
===========================================================================================
The concept of RNA velocity has provided a new way to construct directed dynamics of cellular states by modeling dynamics of pre-mRNAs (unspliced) and mature mRNAs (spliced) from transcriptional dynamic kinetics. The original model of RNA velocity, velocyto, was proposed to estimate RNA velocity based on an assumption mRNA abundance can reach the stationary states (`La Manno et al., Nature, 2018 <https://doi.org/10.1038/s41586-018-0414-6>`_). A substitute model, scVelo, was developed to solve ordinary differential equations (ODEs) of RNA velocity for mRNAs that switch to repression before reaching the stationary state (`Bergen *et al. <https://doi.org/10.1038/s41587-020-0591-3>`_).
.

In our algorithm. We fed the spliced and unspliced mRNA of each cell into cellDancer and predict cell-specific transcription, splicing, and degradation rates. To training cellDancer, we minimized a loss function, which sufficiently measures the similarity between predicted and observed velocity vectors.

The next section described how does the loss decrease.

Understand the decreasing of loss
--------------------------------------
The figure of Sulf2 in the pancreatic endocrinogenesis data shows how the DNN (deep neural network) was trained step by step. During minimizing loss function, the predicted velocity of Sulf2 gradually fits to the observed spliced and unspliced mRNAs.

.. image:: images/training_progress1.png
   :width: 600

Below are more examples showing how the model changes during the training.

.. image:: images/training_progress2.png
   :width: 600