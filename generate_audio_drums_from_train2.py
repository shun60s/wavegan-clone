# -*- coding: utf-8 -*-
"""

-----------------------------------------------------------------------------
Copyright (c) 2019 Christopher Donahue

Please see LICENSE-wavegan.txt.
-----------------------------------------------------------------------------
model.ckpを使ってwavをgenerateする。

change 2021-9-15
1チャンネルのもとの信号に、更に、1チャンネルのラベル（WAVそのもの、代表）を追加して、
Conditional wave GAN を実験するためのもの。　


"""

import tensorflow as tf
import PIL.Image
import numpy as np
import librosa
import loader2
import sys
#from IPython.display import display, Audio
from matplotlib import pyplot as plt


# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph('./train/infer/infer.meta')
graph = tf.get_default_graph()
sess = tf.InteractiveSession()

###
### Please change model.ckpt number to actual one in train diectory. ###
#saver.restore(sess, './train/model.ckpt-417')
saver.restore(sess, './train/model.ckpt-0')


label_dim= loader2.get_label(None)
print ("label_dim=", label_dim)
#



# make one hot order
ngenerate = label_dim*2
indices= np.arange(0,ngenerate,1)
label_one_hot_np=np.identity(label_dim, dtype=np.float32)[indices % label_dim]


# Create ngenerate random latent vectors z
_z = (np.random.rand(ngenerate, 100-label_dim) * 2.) - 1
_z = np.concatenate([_z, label_one_hot_np], axis=1)


# Synthesize G(z)
z = graph.get_tensor_by_name('z:0')

G_z = graph.get_tensor_by_name('G_z:0')
_G_z = sess.run(G_z, {z: _z})

#
ndisplay = label_dim * 2
for i in range(ndisplay):
  print('-' * 80)
  print('Example {} - {}'.format(i, i % label_dim))
  
  print (_G_z[i].T.shape )
 
  plt.figure()
  plt.plot(PIL.Image.fromarray(_G_z[i]))
  
  plt.grid()
  plt.axis('tight')
  plt.show()
  
  y=_G_z[i].reshape( len(_G_z[i]) )
  librosa.output.write_wav('gerated_drum_' + str(i) + '-' + str( i % label_dim) + '.wav', y, sr=16000)
