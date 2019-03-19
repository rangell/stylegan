import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from training import misc
from training import dataset
from dnnlib import EasyDict
import tensorflow as tf
from training.training_loop import process_reals


tflib.init_tf()


dataset_args = EasyDict(tfrecord_dir='images1024x1024');


training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **dataset_args)
print(type(training_set))

lod_in = tf.placeholder(tf.float32, name='lod_in', shape=[])

reals, labels = training_set.get_minibatch_tf()
reals = process_reals(reals, lod_in, False, training_set.dynamic_range, [-1,1])

print(reals)


# Load a snapshot from the training results dir
pkl_file = '/mnt/nfs/scratch1/sbrockman/results/00073-sgan-images1024x1024-4gpu/network-snapshot-002361.pkl'
G, D, Gs, E = misc.load_pkl(pkl_file)


#E.print_layers()


# Load pretrained generator
url = 'https://drive.google.com/uc?export=download&id=1XfM3_jfHd9jrfHcPG0Z1iSAZPjVKe50e'
with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G_pre, _D_pre, Gs_pre = pickle.load(f)


# Feed an image into the encoder

#reals = '/mnt/nfs/scratch1/sbrockman/data/00009.png'
#labels = 0

mu, log_var = E.get_output_for(reals, labels, is_Training=False)



eps = tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)
z = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(log_var)), eps))

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
fake_images = Gs_pre.get_output_for(z, labels, is_training=False)
recon_loss = tf.losses.mean_squared_error(reals, fake_images)
          
KL = -0.5 * tf.reduce_sum(1+log_var - tf.square(mu) - tf.exp(log_var), 1)
cost = tf.reduce_mean(recon_loss + KL)

c = tflib.run(cost)

print('total loss: ', c)
