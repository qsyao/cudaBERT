import tensorflow as tf
import numpy    

key_dir = 'QnA'

tf_path = 'tf_checkpoint/' + key_dir + '/model.ckpt-333162' #file ckpt

init_vars = tf.train.list_variables(tf_path)

names = []
arrays = []

for name, shape in init_vars:
    print("Loading TF weight {} with shape {}".format(name, shape))
    array = tf.train.load_variable(tf_path, name)
    names.append(name)
    arrays.append(array)

for i in range(len(names)):
    names[i] = names[i].replace('/', '_')

import ipdb; ipdb.set_trace()

for i in range(len(names)):
    numpy.save('model_npy/' + key_dir + '/' + names[i], arrays[i])