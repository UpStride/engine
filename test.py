import sys
import pytest
import tensorflow as tf

if __name__ == "__main__":
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  # Channel first is the default for the engine
  tf.keras.backend.set_image_data_format('channels_first')
  pytest.main(sys.argv[1:])
