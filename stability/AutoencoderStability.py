from autoencoder.Autoencoder import Autoencoder
from autoencoder.Evaluation import Evaluation
from autoencoder.Dataset import Dataset

from factorisation.Factorisation import Factorisation

from stability.TrainStability import TrainStability
from stability.LossStability import LossStability
from stability.DatasetStability import DatasetStability

from tools.tools import variable_summaries
from tools.tools import summary_folder

import tensorflow as tf


class AutoencoderStability(Autoencoder):
    def __init__(self, parameters, sets):
        super().__init__(parameters=parameters, sets=sets)

        if parameters['stability']['first_learning'] == 'factorisation':
            self.Factorization = Factorisation(factorisation_sets=sets['factorisation'],
                                               sets_parameters=parameters['sets'],
                                               factorisation_parameters=parameters['factorisation'])
            parameters['stability']['rmse'] = self.Factorization.rmse
            parameters['stability']['differences'] = self.Factorization.difference_matrix.copy()
        else:
            self.Autoencoder = Autoencoder(sets=sets, parameters=parameters)
            self.Autoencoder.run_training()
            parameters['stability']['rmse'] = self.Autoencoder.rmse_train
            parameters['stability']['differences'] = self.Autoencoder.difference_matrix.copy()
            del self.Autoencoder

        self.Train_set = DatasetStability(stability_parameters=parameters['stability'],
                                          dataset=sets['autoencoder'][0],
                                          sets_parameters=parameters['sets'])
        self.Validation_set = Dataset(dataset=sets['autoencoder'][1])
        self.Test_set = Dataset(dataset=sets['autoencoder'][2])

        self.Loss = LossStability()

        self.Train = TrainStability(sets_parameters=parameters['sets'],
                                    Train_set=self.Train_set,
                                    batch_size=self.batch_size_train,
                                    learning_decay=self.learning_decay,
                                    learning_rate0=self.learning_rate0)

        self.Evaluation = Evaluation(sets_parameters=parameters['sets'],
                                     batch_size_evaluate=self.batch_size_evaluate,
                                     Train_set=self.Train_set)

    def run_training(self):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            x_sparse = tf.sparse_placeholder(dtype=tf.float32, name='x_sparse')
            target = tf.placeholder(dtype=tf.float32, name='target')
            coefficients = tf.placeholder(dtype=tf.float32, name='coefficients')

            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

            prediction = self.Train.inference(x_sparse=x_sparse,
                                              target=target,
                                              hidden1_units=self.hidden1_units)

            loss = self.Loss.full_l2_loss_stability(target=target,
                                                    prediction=prediction,
                                                    regularisation=self.regularisation,
                                                    coefficients=coefficients,
                                                    weights_list=tf.trainable_variables())

            train_op = self.Train.training(loss=loss,
                                           optimiser=tf.train.GradientDescentOptimizer(learning_rate=learning_rate))

            square_error = self.Evaluation.square_error(prediction, target)

            # variable_summaries(loss, 'loss/')
            # variable_summaries(learning_rate, 'learning_rate/')
            # summary_op = tf.merge_all_summaries()
            # summary_writer = tf.train.SummaryWriter(summary_folder('logs'), sess.graph)

            server = tf.train.Server.create_local_server()

            init = tf.initialize_all_variables()
            sess = tf.Session(target=server.target)
            sess.run(init)

            for step in range(self.nb_steps):
                epoch = float((step // self.epoch_steps))
                self.Train.learning_rate_update(epoch=epoch)
                feed_dict = self.Train.fill_feed_dict_train_stability(target=target,
                                                                      x_sparse=x_sparse,
                                                                      coefficients=coefficients,
                                                                      learning_rate=learning_rate)
                sess.run(train_op,
                         feed_dict=feed_dict)

                # summary_str = sess.run(summary_op,
                #                        feed_dict=feed_dict)
                # summary_writer.add_summary(summary_str, step)
                # summary_writer.flush()

            print('epoch ' + str(epoch))

            if not self.is_test:
                print('Validation Data Eval:')
                self.rmse = self.Evaluation.rmse(sess=sess,
                                                 square_error_batch=square_error,
                                                 x_sparse=x_sparse,
                                                 target=target,
                                                 data_set=self.Validation_set,
                                                 is_train=False)
            else:
                print('Test Data Eval:')
                self.rmse = self.Evaluation.rmse(sess=sess,
                                                 square_error_batch=square_error,
                                                 x_sparse=x_sparse,
                                                 target=target,
                                                 data_set=self.Test_set,
                                                 is_train=False)
