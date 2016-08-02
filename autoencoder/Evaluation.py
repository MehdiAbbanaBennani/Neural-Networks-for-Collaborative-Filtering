import tensorflow as tf
import math
import numpy as np
from tools import global_parameters
from tools import to_dense
from tools import count


class Evaluation(object):
    def __init__(self, batch_size_evaluate, database_id, Train_set):
        self.batch_size_evaluate = batch_size_evaluate
        self.Train_set = Train_set
        self.nb_users, self.nb_movies = global_parameters(database=database_id)[0:2]

    @staticmethod
    def mini_batch_rmse(estimated, target):
        with tf.name_scope('evaluation'):
            with tf.control_dependencies([tf.assert_equal(count(tf.to_int32(target) - tf.to_int32(target)), 0.)]):
                squared_difference = tf.pow(estimated - target, 2, name='squared_difference')
                square_error = tf.reduce_sum(squared_difference, name='summing_square_errors')
                square_error = tf.to_float(square_error)
                mse = tf.truediv(square_error, count(target), name='meaning_error')
                rmse = tf.sqrt(mse)
                return rmse

    @staticmethod
    def square_error(estimated, target):
        with tf.name_scope('evaluation'):
            with tf.control_dependencies([tf.assert_equal(count(tf.to_int32(target) - tf.to_int32(target)), 0.)]):
                tf.assert_equal(count(tf.cast(target - estimated, tf.int32)), 0.)
                squared_difference = tf.pow(estimated - target, 2, name='squared_difference')
                square_error = tf.reduce_sum(squared_difference, name='summing_square_errors')
                square_error = tf.to_float(square_error)
                return square_error

    def rmse(self, sess, square_error_batch, x_sparse, target, data_set, is_train): #Untested
        square_error = 0
        num_examples = 0
        self.Train_set.reset('rmse')
        data_set.reset('rmse')
        steps_per_epoch = data_set.size // self.batch_size_evaluate

        for step in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict_mini_batch(data_set=data_set,
                                                       x_sparse=x_sparse,
                                                       target=target,
                                                       is_train=is_train)
            num_examples += sess.run(count(target), feed_dict=feed_dict)
            square_error += sess.run(square_error_batch, feed_dict=feed_dict)
        mean_square_error = square_error / num_examples
        rmse = math.sqrt(mean_square_error)
        rmse *= 2
        return rmse

    def fill_feed_dict_mini_batch(self, data_set, target, x_sparse, is_train): # Untested
        x_sparse_indices, x_sparse_values = self.Train_set.next_batch(self.batch_size_evaluate, 'rmse')
        if is_train:
            target_indices = x_sparse_indices
            target_values = x_sparse_values
        else:
            target_indices, target_values = data_set.next_batch(self.batch_size_evaluate, 'rmse')
        shape = np.array([self.batch_size_evaluate, self.nb_movies], dtype=np.int64)
        feed_dict = {
            target: to_dense(target_indices, target_values, shape),
            x_sparse: (x_sparse_indices, x_sparse_values, shape),
        }
        return feed_dict


