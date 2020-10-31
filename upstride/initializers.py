from upstride.type1.tf.keras.initializers import init_aliases_dict as init_type1
from upstride.type2.tf.keras.initializers import init_aliases_dict as init_type2

class InitializersFactory():
  def __init__(self):
    self.init_types = {1: init_type1, 2: init_type2}

    
  def is_custom_init(self, name):
    for k in list(self.init_types.keys()):
      if name in list(self.init_types[k]):
        return True
    return False

  def get_initializer(self, name, upstride_type):
    if name in list(self.init_types[upstride_type].keys()):
      return self.init_types[upstride_type][name]
    else:
      raise ValueError(f'Custom initializers {name} not supported for upstride_type = {upstride_type}')
