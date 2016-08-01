from fs.osfs import OSFS
import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf


def global_parameters(database):
    if database == 0:
        nb_movies = int(1682)
        nb_users = 943
        data_set_size = 1000209
        data_file = '/home/sequel/Pycharm project/Autoencoder/Database/ratings100K.csv'
    elif database == 1:
        nb_movies = int(3952)
        nb_users = 6040
        data_set_size = 1000209
        data_file = "/home/sequel/Pycharm project/Autoencoder_Stability/Databases/ratings1M.csv"
    elif database == 2:
        nb_movies = int(10681)
        nb_users = 71567
        data_set_size = 10000054
        data_file = "Databases/ratings10M.csv"
    else:
        print('No such dataset')
    return nb_users, nb_movies, data_set_size, data_file


def summary_folder(name):
    logdir = '/home/mehdi/PycharmProjects/Autoencoder/Neural-Network-Collaborative-Filtering/tmp/' + name
    folder = OSFS(logdir)
    test_n = len(list(n for n in folder.listdir() if n.startswith('test')))
    return logdir + "/test" + str(test_n + 1)


def to_dense(indices, values, shape):
    dense_array = csr_matrix((values, (indices[:, 0], indices[:, 1])), shape=shape).toarray()
    return dense_array.astype(np.float32)


def count(dense_tensor):
    with tf.name_scope('non_zero_count'):
        indicator = tf.cast(dense_tensor, tf.bool, name='bool_casting')
        indicator_float = tf.cast(indicator, tf.float32)
        size = tf.reduce_sum(indicator_float)
    return size


def log_folder():
    log_dir = '/home/mehdi/PycharmProjects/Autoencoder/Neural-Network-Collaborative-Filtering/Experiments/logs/'
    folder = OSFS(log_dir)
    test_n = len(list(n for n in folder.listdir() if n.startswith('test')))
    return log_dir + "/test" + str(test_n + 1)


def variable_summaries(var, name):
  with tf.name_scope(name + 'summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary(name + 'mean', mean)
    with tf.name_scope(name + 'stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary(name + 'sttdev', stddev)
    tf.scalar_summary(name + 'max', tf.reduce_max(var))
    tf.scalar_summary(name + 'min', tf.reduce_min(var))
    tf.histogram_summary(name, var)
