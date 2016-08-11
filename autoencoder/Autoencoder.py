from autoencoder.Dataset import Dataset
from autoencoder.Train import Train
from autoencoder.Loss import Loss
from autoencoder.Evaluation import Evaluation
from tools.tools import summary_folder
from tools.tools import variable_summaries
from tools.tools import global_parameters

import tensorflow as tf

from tensorflow.python.client import timeline

import time


class Autoencoder(object):
    def __init__(self, parameters, sets):

        self.database = parameters['sets']['database_id']
        self.hidden1_units = parameters['autoencoder']['hidden1_units']
        self.regularisation = parameters['autoencoder']['regularisation']
        self.learning_rate0 = parameters['autoencoder']['learning_rate0']
        self.learning_decay = parameters['autoencoder']['learning_decay']
        self.batch_size_evaluate = parameters['autoencoder']['batch_size_evaluate']
        self.batch_size_train = parameters['autoencoder']['batch_size_train']
        self.is_test = parameters['autoencoder']['is_test']
        self.nb_users, self.nb_movies = global_parameters(sets_parameters=parameters['sets'])[0:2]

        self.difference_matrix = 0
        self.rmse = 0
        self.rmse_train = 0

        self.epoch_steps = int(self.nb_users / self.batch_size_train)
        self.nb_steps = parameters['autoencoder']['nb_epoch'] * self.epoch_steps

        self.Train_set = Dataset(sets['autoencoder'][0])
        self.Validation_set = Dataset(sets['autoencoder'][1])
        self.Test_set = Dataset(sets['autoencoder'][2])

        self.Loss = Loss()

        self.Train = Train(sets_parameters=parameters['sets'],
                           Train_set=self.Train_set,
                           batch_size=self.batch_size_train,
                           learning_decay=self.learning_decay,
                           learning_rate0=self.learning_rate0)

        self.Evaluation = Evaluation(sets_parameters=parameters['sets'],
                                     batch_size_evaluate=self.batch_size_evaluate,
                                     Train_set=self.Train_set)

    def run_training(self):
        with tf.device('/cpu:0'):
            x_sparse = tf.sparse_placeholder(dtype=tf.float32, name='x_sparse')
            target = tf.placeholder(dtype=tf.float32, name='target')
            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

            prediction = self.Train.inference(x_sparse=x_sparse,
                                              target=target,
                                              hidden1_units=self.hidden1_units)

            difference = target - prediction

            loss = self.Loss.full_l2_loss(target=target,
                                          prediction=prediction,
                                          regularisation=self.regularisation,
                                          weights_list=tf.trainable_variables())

            train_op = self.Train.training(loss=loss,
                                           optimiser=tf.train.GradientDescentOptimizer(learning_rate=learning_rate))

            square_error = self.Evaluation.square_error(prediction, target)

        variable_summaries(loss, 'loss/')
        variable_summaries(learning_rate, 'learning_rate/')
        # summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        # summary_writer = tf.train.SummaryWriter(summary_folder('logs'), sess.graph)
        sess.run(init)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        start = time.time()

        for step in range(self.nb_steps):
            epoch = float((step // self.epoch_steps))
            self.Train.learning_rate_update(epoch=epoch)
            feed_dict = self.Train.fill_feed_dict_train(target=target,
                                                        x_sparse=x_sparse,
                                                        learning_rate=learning_rate)
            sess.run(train_op,
                     feed_dict=feed_dict,
                     options=run_options,
                     run_metadata=run_metadata
                     )

            # summary_str = sess.run(summary_op,
            #                        feed_dict=feed_dict,
            #                        options=run_options,
            #                        run_metadata=run_metadata)
            # summary_writer.add_summary(summary_str, step)
            # summary_writer.flush()

            # if step == (self.epoch_steps - 1):

        end = time.time()
        print('training time:' + str(end - start))

        print('epoch ' + str(epoch))

        start = time.time()

        if not self.is_test:
            print('Validation Data Eval:')
            self.rmse = self.Evaluation.rmse(sess=sess,
                                             square_error_batch=square_error,
                                             x_sparse=x_sparse,
                                             target=target,
                                             data_set=self.Validation_set,
                                             is_train=False)
            print(self.rmse)
        else:
            print('Test Data Eval:')
            self.rmse = self.Evaluation.rmse(sess=sess,
                                             square_error_batch=square_error,
                                             x_sparse=x_sparse,
                                             target=target,
                                             data_set=self.Test_set,
                                             is_train=False)
            print(self.rmse)

        end = time.time()
        print('Evaluation time:' + str(end - start))

        start = time.time()

        print('Train Data Eval:')
        self.rmse_train = self.Evaluation.rmse(sess=sess,
                                               square_error_batch=square_error,
                                               x_sparse=x_sparse,
                                               target=target,
                                               data_set=self.Train_set,
                                               is_train=True)
        print(self.rmse_train)

        self.difference_matrix = self.Evaluation.differences(difference_op=difference,
                                                              data_set=self.Train_set,
                                                              sess=sess,
                                                              is_train=True,
                                                              x_sparse=x_sparse,
                                                              target=target)

        end = time.time()
        print('differences evaluation time ' + str(end - start))

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)