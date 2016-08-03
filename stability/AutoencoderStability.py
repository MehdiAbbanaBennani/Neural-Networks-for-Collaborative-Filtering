from autoencoder.Autoencoder import Autoencoder
from autoencoder.Evaluation import Evaluation
from autoencoder.Dataset import Dataset

from factorisation.Factorisation import Factorisation

from stability.TrainStability import TrainStability
from stability.LossStability import LossStability
from stability.DatasetStability import DatasetStability

from tools import variable_summaries
from tools import summary_folder

import tensorflow as tf


class AutoencoderStability(Autoencoder):
    def __init__(self, autoencoder_parameters, autoencoder_sets, factorisation_sets,
                 sets_parameters, factorisation_parameters, stability_parameters):
        super().__init__(autoencoder_parameters, autoencoder_sets, sets_parameters)

        self.Factorization = Factorisation(factorisation_sets=factorisation_sets,
                                           sets_parameters=sets_parameters,
                                           factorisation_parameters=factorisation_parameters)
        stability_parameters['rmse'] = self.Factorization.rmse
        stability_parameters['differences'] = self.Factorization.difference_matrix

        self.Train_set = DatasetStability(stability_parameters=stability_parameters,
                                          dataset=autoencoder_sets[0],
                                          sets_parameters=sets_parameters)
        self.Validation_set = Dataset(dataset=autoencoder_sets[1])
        self.Test_set = Dataset(dataset=autoencoder_sets[2])

        self.Loss = LossStability()

        self.Train = TrainStability(database=self.database,
                                    Train_set=self.Train_set,
                                    batch_size=self.batch_size_train,
                                    learning_decay=self.learning_decay,
                                    learning_rate0=self.learning_rate0)

        self.Evaluation = Evaluation(database_id=self.database,
                                     batch_size_evaluate=self.batch_size_evaluate,
                                     Train_set=self.Train_set)

    def run_training(self):
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

            variable_summaries(loss, 'loss/')
            variable_summaries(learning_rate, 'learning_rate/')
            summary_op = tf.merge_all_summaries()
            init = tf.initialize_all_variables()
            sess = tf.Session()
            summary_writer = tf.train.SummaryWriter(summary_folder('logs'), sess.graph)
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

                summary_str = sess.run(summary_op,
                                       feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                if step % (1 * self.epoch_steps) == 0:
                    print('epoch ' + str(epoch))
                    # saver.save(sess, summary_folder('checkpoints'), global_step=step)

                    # print('Training Data Eval:')
                    # print(Evaluation.rmse(sess,
                    #                       square_error_batch=square_error,
                    #                       x_sparse=x_sparse,
                    #                       target=target,
                    #                       data_set=Train_set,
                    #                       is_train=True))

                    print('Validation Data Eval:')
                    self.rmse = self.Evaluation.rmse(sess,
                                          square_error_batch=square_error,
                                          x_sparse=x_sparse,
                                          target=target,
                                          data_set=self.Validation_set,
                                          is_train=False)
                    print(self.rmse)

                    # print('Test Data Eval:')
                    # print(Evaluation.rmse(sess,
                    #                 square_error_batch=square_error,
                    #                 x_sparse=x_sparse,
                    #                 target=target,
                    #                 data_set=Test_set))
