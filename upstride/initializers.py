from upstride.type1.tf.keras.initializers import init_aliases_dict as init_type1
from upstride.type2.tf.keras.initializers import init_aliases_dict as init_type2

class InitializersFactory():
  def __init__(self):
    self.init_types = {1: init_type1, 2: init_type2}
    
  def is_custom_init(self, name):
    if name in list(self.init_types.keys()):
      return True
    return False

  def get_initializer(self, name, upstride_type):
    if upstride_type == 1 and name in list(init_type1.keys()):
      return self.init_type1[name]
    elif upstride_type == 2 and name in list(init_type2.keys()):
      return self.init_type2[name]
    else:
      raise ValueError(f'Custom initializers {name} not supported for upstride_type = {upstride_type}')