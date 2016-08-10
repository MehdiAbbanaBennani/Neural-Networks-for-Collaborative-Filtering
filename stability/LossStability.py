import tensorflow as tf
from autoencoder.Loss import Loss

from tools.tools import count
from tools.tools import indicator


class LossStability(Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def loss_l2_stability(estimated, target, coefficients):
        """estimated and target are dense tensors"""
        with tf.name_scope('l2_loss'):
            # with tf.control_dependencies([tf.assert_equal(count(indicator(target) - indicator(estimated)), 0.)]):
                # tf.assert_equal(count(indicator(target) - indicator(coefficients)), 0.)
            print('check assertion')
            # TODO check assertion
            squared_difference = tf.pow(estimated - target, 2, name='squared_difference')
            squared_difference_stable = tf.mul(squared_difference, coefficients,
                                               name='stability_coefficients_product')
            loss = tf.reduce_sum(squared_difference_stable, name='summing_square_errors_with_stability')
            return loss

    def full_l2_loss_stability(self, regularisation, weights_list, prediction, target, coefficients): # Unchecked
        with tf.name_scope('loss'):
            loss = tf.add(self.l2_regularisation(regularisation, weights_list),
                          self.loss_l2_stability(estimated=prediction, target=target, coefficients=coefficients),
                          name='full_loss')
        return loss
