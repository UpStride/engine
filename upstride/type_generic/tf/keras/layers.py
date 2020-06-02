from .... import generic_layers
from .... generic_layers import *


generic_layers.blade_indexes = None
generic_layers.geometrical_def = None

def define_ga(a, b, c, blades):
    generic_layers.blade_indexes = blades
    generic_layers.geometrical_def = (a, b, c)
