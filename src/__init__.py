# will not be used in usual, but if we want to make it a package, this .py file is needed to exist.
# need to write readme to teach user how to use the functions in the package.

from .simulation_pl3 import generate_simulation, load_simulation, show_simus

from .model_pl3 import train, pretrain_model_path

from .analysis import show_details

from .utilities import set_rcParams
