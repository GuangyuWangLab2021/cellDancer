API - cellDancer key applications
===========================================================================================

Import pandas, cellDancer, and plotting modules as::
   
   import pandas as pd
   import celldancer as cd
   import celldancer.plotting as cdplt

After loaded the data (``pd.read_csv``), the prediction could be done by ``cd.velocity_estimation.velocity``. The projection of velocity to embedding space could be calculated by ``cd.compute_cell_velocity.compute`` and visualized by ``cd.cell.scatter_cell``. The pseudotime could be calculated by ``cd.pseudo_time.pseudo_time`` and visualized by ``cell.scatter_cell`` or ``gene.scatter_gene``, the UMAP based one kinetic parameters could be calculated by ``cd.embedding_kinetic_para.embedding`` and visualized by ``cell.plot_kinetic_para``.


Toolkit functions
-------------------
Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: celldancer
.. rubric:: Functions
.. autosummary::

   utilities.adata_to_raw_with_embed

Velocity estimation and analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: celldancer
.. rubric:: Functions

.. autosummary::

   velocity_estimation.velocity
   compute_cell_velocity.compute
   pseudo_time.pseudo_time
   embedding_kinetic_para.embedding


Plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: celldancer.plotting
.. rubric:: Functions

.. autosummary::

   gene.scatter_gene
   cell.scatter_cell
   cell.plot_kinetic_para
   graph.graph



