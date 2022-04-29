.. toolkit documentation master file, created by
   sphinx-quickstart on Wed Feb  9 17:10:01 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

API - CellDancer key applications
===========================================================================================
celldancer is a scalable toolkit for RNA velocity analysis in single cells, based on XXX.

This is the most important part of a documentation theme. If you like
the general look of the theme, please make sure that it is possible to
easily navigate through this sample documentation.

Ideally, the pages listed below should also be reachable via links
somewhere else on this page (like the sidebar, or a topbar). If they are
not, then this theme might need additional configuration to provide the
sort of site navigation that's necessary for "real" documentation.



Toolkit functions
-------------------


celldancer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: celldancer
.. rubric:: Functions

.. autosummary::
   :toctree: 

   velocity_estimation.train


plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: plotting
.. rubric:: Functions

.. autosummary::
   :toctree: 

   gene.scatter_gene
   compute_cell_velocity.compute_cell_velocity
   pseudo_time.pseudo_time
   cell.scatter_cell
   cell.calculate_para_umap
   cell.plot_para_umap



