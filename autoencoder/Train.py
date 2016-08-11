import tensorflow as tf
import math
import numpy as np

from tools.tools import global_parameters
from tools.tools import to_dense
from tools.tools import variable_summaries


class Train(object):
    def __init__(self, sets_parameters, Train_set, batch_size, learning_rate0, learning_decay):
        self.nb_users, self.nb_movies = global_parameters(sets_parameters=sets_parameters)[0:2]
        self.Train_set = Train_set
        self.batch_size = batch_size
        self.learning_rate0 = learning_rate0
        self.learning_rate = learning_rate0
        self.learning_decay = learning_decay

    @staticmethod
    def nn_layer(input, input_size, output_size, name):
        with tf.name_scope('hidden' + name + '/'):
            weights_var = 1 / math.sqrt(input_size)
            weights = tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                    minval=-weights_var,
                                                    maxval=weights_var),
                                  name='weights')
            biases = tf.Variable(tf.zeros([output_size]),
                                 name='biases')

            preactivation = tf.add(tf.sparse_tensor_dense_matmul(input, weights), biases, name='preactivation')

            activation = tf.nn.tanh(preactivation, name='activation')

            variable_summaries(weights, name='Layer' + name + '/weights')
            variable_summaries(biases, name='Layer' + name + '/biases')
            variable_summaries(preactivation, name='Layer' + name + '/preactivation')
            variable_summaries(activation, name='Layer' + name + '/activation')

            return activation

    @staticmethod
    def nn_layer2(input, input_size, output_size, name):
        with tf.name_scope('hidden' + name + '/'):
            weights_var = 1 / math.sqrt(input_size)
            weights = tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                    minval=-weights_var,
                                                    maxval=weights_var),
                                  name='weights')
            biases = tf.Variable(tf.zeros([output_size]),
                                 name='biases')

            preactivation = tf.add(tf.matmul(input, weights), biases, name='preactivation')
            activation = tf.nn.tanh(preactivation, name='activation')

            variable_summaries(weights, name='Layer' + name + '/weights')
            variable_summaries(biases, name='Layer' + name + '/biases')
            variable_summaries(preactivation, name='Layer' + name + '/preactivation')
            variable_summaries(activation, name='Layer' + name + '/activation')

            return activation

    @staticmethod
    def mask(to_mask, mask):
        with tf.name_scope('masking'):
            indicator = tf.cast(mask, tf.bool, name='to_bool')
            float_indicator = tf.to_float(indicator, name='to_float')
            masked_output = tf.mul(to_mask, float_indicator, name='masking')
        return masked_output

    def inference(self, x_sparse, hidden1_units, target):
        activation1 = self.nn_layer(input=x_sparse,
                                    input_size=self.nb_movies,
                                    output_size=hidden1_units,
                                    name='1')
        activation2 = self.nn_layer2(input=activation1,
                                    input_size=hidden1_units,
                                    output_size=self.nb_movies,
                                    name='2')
        masked_activation2 = self.mask(to_mask=activation2, mask=target)
        return masked_activation2

    @staticmethod
    def training(loss, optimiser):
        tf.scalar_summary(loss.op.name, loss)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimiser.minimize(loss=loss, global_step=global_step)
        return train_op

    def fill_feed_dict_train(self, target, x_sparse, learning_rate):
        x_sparse_indices, x_sparse_values = self.Train_set.next_batch(self.batch_size, 'user')
        shape = np.array([self.batch_size, self.nb_movies], dtype=np.int64)
        feed_dict = {
            target: to_dense(x_sparse_indices, x_sparse_values, shape),
            x_sparse: (x_sparse_indices, x_sparse_values, shape),
            learning_rate: self.learning_rate
        }
        return feed_dict

    def learning_rate_update(self, epoch):
        self.learning_rate = self.learning_rate0 / (1 + self.learning_decay * epoch)