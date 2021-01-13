import tensorflow as tf 
import argparse
from collections import defaultdict
import os, csv
import numpy as np

from upstride.initializers import InitializersFactory
from src.conv_from_dcn import ComplexConv2D
from upstride.type1.tf.keras import layers
from upstride.test_batchnorm import Channel2Batch, Batch2Channel

tf.keras.backend.set_image_data_format('channels_first')
parser = argparse.ArgumentParser("compare_init")
parser.add_argument('--DCN_ours', action='store_true', default=False,help='run DCN our code')
args = parser.parse_args()

init_factory = InitializersFactory()
def main(args):
  conv_args = {
    "filters": 3,
    "kernel_size": 1,
    # "init_criterion": 'he',
    "kernel_initializer": "complex_independent" if not args.DCN_ours else init_factory.get_initializer('he_ind',1),
    "data_format": "channels_first",
    "use_bias": False
  }
  subset = 1 if args.DCN_ours else 0 # for picking the kernel:0 name in weights 
  batch_size = 16
  epochs = 1

  def get_model(seed = 42):
    inputs = tf.keras.layers.Input(shape=(6,3,3))
    if args.DCN_ours:
      x = Channel2Batch()(inputs)
      x = layers.Conv2D(**conv_args,name="conv_1")(x)
      x = layers.Conv2D(**conv_args,name="conv_2")(x)
      x = layers.Conv2D(**conv_args,name="conv_3")(x)
      x = layers.Conv2D(**conv_args,name="conv_4")(x)
      x = Batch2Channel()(x)
    else:
      x = ComplexConv2D(**conv_args,seed=seed,name="conv_1")(inputs)
      x = ComplexConv2D(**conv_args,seed=seed,name="conv_2")(x)
      x = ComplexConv2D(**conv_args,seed=seed,name="conv_3")(x)
      x = ComplexConv2D(**conv_args,seed=seed,name="conv_4")(x)
    x = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    # model.summary()
    return model

  list_of_batches = []
  for i in range(1, 11): 
    list_of_batches.append(tf.ones((batch_size,6,3,3)) * i)
  X = tf.concat(list_of_batches,axis=0)
  y = tf.ones((batch_size * len(list_of_batches) ,64*3*3))
  dataset = tf.data.Dataset.from_tensor_slices((X, y))
  dataset = dataset.batch(batch_size)

  loss = tf.keras.losses.MeanSquaredError()
  optimizer = tf.keras.optimizers.Adam(lr=0.01)

  weight_dict = defaultdict(lambda: defaultdict(list))
  for i in range(50):
    mod = get_model()
    for weight in mod.weights:
      weight_name = weight.name.split("/")[subset]
      weight_value = weight.numpy()
      mean = np.mean(weight_value)
      std = np.std(weight_value)
      weight_dict[weight_name]["mean"].append(mean) 
      weight_dict[weight_name]["std"].append(std)
    tf.keras.backend.clear_session() 
  # print(weight_dict[weight_name])
  for conv in weight_dict.keys():
    print(conv, np.mean(weight_dict[conv]["mean"]),np.mean(weight_dict[conv]["std"]))

  # model = get_model()
  # weight_dict = {}
  # for e in range(epochs):
  #   # print(f"Epoch number {e}/{self.epochs}")
  #   for step, (x_batch, y_batch) in enumerate(dataset):
  #     with tf.GradientTape() as tape:
  #       y_hat = model(x_batch, training=True)
  #       loss_value = loss(y_batch, y_hat)
  #     grads = tape.gradient(loss_value, model.trainable_weights)
  #     optimizer.apply_gradients(zip(grads,model.trainable_weights))

  # for weight in model.weights:
  #   weight_name = weight.name.split('/')[subset]
  #   weight_dict[weight_name] = weight.numpy() #picking the first value from the weight (values ares same)
  
  # print(weight_dict)
  # print(weight_dict['kernel:0'].shape)
  # mean_one_conv = np.mean(weight_dict['kernel:0'])
  # std_one_conv = np.std(weight_dict['kernel:0'])
  # print(mean_one_conv)
  # print(std_one_conv)

  # os.makedirs("DCN_ours",exist_ok=True)
  # os.makedirs("DCN_source",exist_ok=True)

  # for f in weight_dict.keys():
  #   filename = f"DCN_source/{f}.csv" if not args.DCN_ours else f"DCN_ours/{f}.csv"
  #   with open (filename, 'w') as fi:
  #     csv_writer = csv.writer(fi)
  #     csv_writer.writerow(['Step','Weight_value'])
  #     csv_writer.writerows(weight_dict[f])

if __name__ == "__main__":
  main(args)