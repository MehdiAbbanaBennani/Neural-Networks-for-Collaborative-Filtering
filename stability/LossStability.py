import tensorflow as tf
from tools import count
from tools import indicator


class LossStability(object):
    def __init__(self):
        pass

    @staticmethod
    def loss_l2(estimated, target, coefficients):
        """estimated and target are dense tensors"""
        with tf.name_scope('l2_loss'):
            with tf.control_dependencies([tf.assert_equal(count(indicator(target) - indicator(estimated)), 0.),
                                          tf.assert_equal(count(indicator(target) - indicator(coefficients)), 0.)]):
                squared_difference = tf.pow(estimated - target, 2, name='squared_difference')
                squared_difference_stable = tf.mul(squared_difference, coefficients,
                                                   name='stability_coefficients_product')
                loss = tf.reduce_sum(squared_difference_stable, name='summing_square_errors_with_stability')
                return loss

    @staticmethod # Unchecked
    def l2_regularisation(regularisation, weights_list):
        with tf.name_scope('regularisation'):
            regularizer = tf.contrib.layers.l2_regularizer(regularisation)
            regularisation_penalty = tf.contrib.layers.apply_regularization(regularizer=regularizer,
                                                                            weights_list=weights_list)
        return regularisation_penalty

    def full_l2_loss(self, regularisation, weights_list, prediction, target, coefficients): # Unchecked
        with tf.name_scope('loss'):
            loss = tf.add(self.l2_regularisation(regularisation, weights_list),
                          self.loss_l2(estimated=prediction, target=target, coefficients=coefficients),
                          name='full_loss')
        return loss