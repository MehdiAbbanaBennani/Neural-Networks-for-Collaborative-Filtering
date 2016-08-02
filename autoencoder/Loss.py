import tensorflow as tf
from tools import count
from tools import indicator


class Loss(object):
    def __init__(self):
        pass

    @staticmethod
    def loss_l2(estimated, target):
        """estimated and target are dense tensors"""
        with tf.name_scope('l2_loss'):
            with tf.control_dependencies([tf.assert_equal(count(indicator(target) - indicator(estimated)), 0.)]):
                squared_difference = tf.pow(estimated - target, 2, name='squared_difference')
                loss = tf.reduce_sum(squared_difference, name='summing_square_errors')
                return loss

    @staticmethod # Unchecked
    def l2_regularisation(regularisation, weights_list):
        with tf.name_scope('regularisation'):
            regularizer = tf.contrib.layers.l2_regularizer(regularisation)
            regularisation_penalty = tf.contrib.layers.apply_regularization(regularizer=regularizer,
                                                                            weights_list=weights_list)
        return regularisation_penalty

    def full_l2_loss(self, regularisation, weights_list, prediction, target): # Unchecked
        with tf.name_scope('loss'):
            loss = tf.add(self.l2_regularisation(regularisation, weights_list),
                          self.loss_l2(estimated=prediction, target=target),
                          name='full_loss')
        return loss
