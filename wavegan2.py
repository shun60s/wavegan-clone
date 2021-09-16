# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
Copyright (c) 2019 Christopher Donahue

Please see LICENSE-wavegan.txt.
-----------------------------------------------------------------------------


change 2021-9-15
1チャンネルのもとの信号に、更に、1チャンネルのラベル（WAVそのもの、代表）を追加して、
Conditional wave GAN を実験するためのもの。　

"""

"""
ネットワークは、画像で実績のあるDCGANを真似た作りになっている。


"""


import tensorflow as tf


def conv1d_transpose(
    inputs,
    filters,
    kernel_width,
    stride=4,
    padding='same',
    upsample='zeros'):
  """
  低次元のベクトルからより長い次元のベクトルを作る　アップサンプルする方法を提示している
  DCGAN uses small (5x5), twodimensional filters while WaveGAN uses longer (length-25), one-dimensional filters 
  and a larger upsampling factor. Both strategies have the same number of parameters and numerical operations
  
  
  we use longer one-dimensional filters of length 25 （wavegan_kernel_len=25）,instead of two-dimensional
  オーディオを信号は波長が長いので　画像と比べて長いスパンのフィルターが必要
  filters of size 5x5, and we upsample by a factor of 4 instead of 2 at each layer (Figure 2). We
  """
  if upsample == 'zeros':
    return tf.layers.conv2d_transpose(
        tf.expand_dims(inputs, axis=1),
        filters,
        (1, kernel_width),
        strides=(1, stride),
        padding='same'
        )[:, 0]
  elif upsample == 'nn':
    batch_size = tf.shape(inputs)[0]
    _, w, nch = inputs.get_shape().as_list()

    x = inputs

    x = tf.expand_dims(x, axis=1)
    x = tf.image.resize_nearest_neighbor(x, [1, w * stride])
    x = x[:, 0]

    return tf.layers.conv1d(
        x,
        filters,
        kernel_width,
        1,
        padding='same')
  else:
    raise NotImplementedError


"""
  Input: [None, 100]
  Output: [None, slice_len, 1]
"""
def WaveGANGenerator(
    z,
    slice_len=16384,
    nch=1,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    labels=False,  # add
    train=False):
  assert slice_len in [16384, 32768, 65536]
  batch_size = tf.shape(z)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

  # FC and reshape for convolution
  # [100] -> [16, 1024]
  dim_mul = 16 if slice_len == 16384 else 32
  output = z
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, 4 * 4 * dim * dim_mul)
    output = tf.reshape(output, [batch_size, 16, dim * dim_mul])
    output = batchnorm(output)
  output = tf.nn.relu(output)
  dim_mul //= 2

  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_0'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)
  dim_mul //= 2

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_1'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)
  dim_mul //= 2

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_2'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)
  dim_mul //= 2

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3'):
    output = conv1d_transpose(output, dim * dim_mul, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  """
  
  オーディオの波形を処理する1単位slice_lenが16384,  つまり16KHzサンプリングで長さ約1秒 
  """
  # add
  if nch == 2 and labels:
    nch0=1
  else:
    nch0=nch
  # end of add
  # chg from nch to nch0, below
     
  if slice_len == 16384:
    # Layer 4
    # [4096, 64] -> [16384, nch0]
    with tf.variable_scope('upconv_4'):
      output = conv1d_transpose(output, nch0, kernel_len, 4, upsample=upsample)
    output = tf.nn.tanh(output)
  elif slice_len == 32768:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    with tf.variable_scope('upconv_4'):
      output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
      output = batchnorm(output)
    output = tf.nn.relu(output)

    # Layer 5
    # [16384, 64] -> [32768, nch0]
    with tf.variable_scope('upconv_5'):
      output = conv1d_transpose(output, nch0, kernel_len, 2, upsample=upsample)
    output = tf.nn.tanh(output)
  elif slice_len == 65536:
    # Layer 4
    # [4096, 128] -> [16384, 64]
    with tf.variable_scope('upconv_4'):
      output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
      output = batchnorm(output)
    output = tf.nn.relu(output)

    # Layer 5
    # [16384, 64] -> [65536, nch0]
    with tf.variable_scope('upconv_5'):
      output = conv1d_transpose(output, nch0, kernel_len, 4, upsample=upsample)
    output = tf.nn.tanh(output)

  # Automatically update batchnorm moving averages every time G is used during training
  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
    if slice_len == 16384:
      assert len(update_ops) == 10
    else:
      assert len(update_ops) == 12
    with tf.control_dependencies(update_ops):
      output = tf.identity(output)

  return output


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


"""
オーディオ特融の繰り返しパターンが存在するときに最適化問題解決のため、
discriminatorだけ、列に位置を移動する（パターン性を少し無くす効果があるのか？）


Generative image models that upsample by transposed
convolution (such as DCGAN) are known to produce
characteristic “checkerboard” artifacts in images (Odena
et al., 2016). Periodic patterns are less common in images
(Section 3.1), and thus the discriminator can learn to
reject images that contain them.

To prevent the discriminator from learning such a solution,
we propose the phase shuffle operation with hyperparameter
n.
"""



def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x


"""
  Input: [None, slice_len, nch]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(
    x,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0):
  batch_size = tf.shape(x)[0]
  slice_len = int(x.get_shape()[1])
  
  """
  We modify the discriminator in a similar way, using length-25 filters in one dimension 
  and increasing stride from 2 to 4.
  """
  
  """
  wavegan_batchnorm=False,   BATCH NORMALIZEはオフ
  """
  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  """
  wavegan_disc_phaseshuffle=2　discriminatorで繰り返しパターンを避けるため、列の移動する。

  """
  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  # Layer 0
  # [16384, 1] -> [4096, 64]
  output = x
  with tf.variable_scope('downconv_0'):
    output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME')
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('downconv_1'):
    output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('downconv_2'):
    output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('downconv_3'):
    output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('downconv_4'):
    output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)

  if slice_len == 32768:
    # Layer 5
    # [32, 1024] -> [16, 2048]
    with tf.variable_scope('downconv_5'):
      output = tf.layers.conv1d(output, dim * 32, kernel_len, 2, padding='SAME')
      output = batchnorm(output)
    output = lrelu(output)
  elif slice_len == 65536:
    # Layer 5
    # [64, 1024] -> [16, 2048]
    with tf.variable_scope('downconv_5'):
      output = tf.layers.conv1d(output, dim * 32, kernel_len, 4, padding='SAME')
      output = batchnorm(output)
    output = lrelu(output)

  """
  最後のconv層が [16, 2048]になるように調整して、
  Flatten して、desnseで判定かな。
  
  """

  # Flatten
  output = tf.reshape(output, [batch_size, -1])

  # Connect to single logit
  with tf.variable_scope('output'):
    output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
