import tensorflow as tf
import math
import numpy as np
from tools.tools import global_parameters
from tools.tools import to_dense
from tools.tools import count
from tools.tools import to_sparse2


class Evaluation(object):
    def __init__(self, batch_size_evaluate, sets_parameters, Train_set):
        self.batch_size_evaluate = batch_size_evaluate
        self.Train_set = Train_set
        self.nb_users, self.nb_movies = global_parameters(sets_parameters=sets_parameters)[0:2]

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
                                                       is_train=is_train,
                                                       batch_size=self.batch_size_evaluate)
            num_examples += sess.run(count(target), feed_dict=feed_dict)
            square_error += sess.run(square_error_batch, feed_dict=feed_dict)
        mean_square_error = square_error / num_examples
        rmse = math.sqrt(mean_square_error)
        rmse *= 2
        return rmse

    def fill_feed_dict_mini_batch(self, data_set, target, x_sparse, is_train, batch_size): # Untested
        x_sparse_indices, x_sparse_values = self.Train_set.next_batch(batch_size, 'rmse')
        if is_train:
            target_indices = x_sparse_indices
            target_values = x_sparse_values
        else:
            target_indices, target_values = data_set.next_batch(batch_size, 'rmse')
        # TODO check the while
        while np.size(x_sparse_indices) == 0:
            x_sparse_indices, x_sparse_values = self.Train_set.next_batch(batch_size, 'rmse')
            if is_train:
                target_indices = x_sparse_indices
                target_values = x_sparse_values
            else:
                target_indices, target_values = data_set.next_batch(batch_size, 'rmse')

        shape = np.array([batch_size, self.nb_movies], dtype=np.int64)
        feed_dict = {
            target: to_dense(target_indices, target_values, shape),
            x_sparse: (x_sparse_indices, x_sparse_values, shape),
        }
        return feed_dict

    def differences(self, difference_op, sess, data_set, x_sparse, target, is_train):
        self.Train_set.reset('rmse')
        data_set.reset('rmse')
        differences = []

        for step in range(self.nb_users):
            feed_dict = self.fill_feed_dict_mini_batch(data_set=data_set,
                                                       x_sparse=x_sparse,
                                                       target=target,
                                                       is_train=is_train,
                                                       batch_size=1)
            difference = sess.run(difference_op, feed_dict=feed_dict)
            self.update_array(differences, array=difference, row=step)
        assert np.size(differences) == np.size(self.Train_set.ratings.data)
        differences_matrix = to_sparse2(indices=self.Train_set.ratings.indices,
                                        indptr=self.Train_set.ratings.indptr,
                                        values=np.array(differences),
                                        shape=(self.nb_users, self.nb_movies))
        return differences_matrix

    def update_array(self, differences, array, row):
        row_indices = self.Train_set.ratings.getrow(row).indices
        sub_array = array[0][row_indices]
        for r in sub_array:
            differences.append(r)