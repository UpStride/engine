import unittest
import tensorflow as tf
import tensorflow_datasets as tfds

from upstride.type0.tf.keras import layers


class TestUpstride(unittest.TestCase):
  def test_network(self):
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.keras.layers.Input(shape=(3, 32, 32))
    x = layers.TF2Upstride()(inputs)
    x = layers.Conv2D(8, (3, 3), name='test-names')(x)
    # #self.assertEqual(len(x), 1)
    # self.assertEqual(len(x), 2)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(24, (3, 3))(x)
    x = layers.BatchNormalization(center=False, scale=False)(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(50)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(50)(x)
    x = layers.Upstride2TF()(x)
    x = tf.keras.layers.Activation('relu')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_type1.png', show_shapes=True)

    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
      """Normalizes images: `uint8` -> `float32`."""
      return tf.transpose(tf.cast(image, tf.float32) / 255., [2, 0, 1]), label

    ds_train = ds_train.map(normalize_img)
    ds_train = ds_train.batch(128)
    ds_test = ds_test.map(normalize_img)
    ds_test = ds_test.batch(128)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )
    logs = "/tmp/logs/"
    callback = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch='100,120')

    model.fit(ds_train,
              epochs=2,
              validation_data=ds_test,
              callbacks=[callback])


if __name__ == "__main__":
  unittest.main()
