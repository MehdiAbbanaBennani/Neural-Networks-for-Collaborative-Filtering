import tensorflow as tf
from Evaluation import Evaluation
from Loss import Loss
from Train import Train

from Autoencoder.Import import Import
from tools import global_parameters
from tools import summary_folder
from tools import variable_summaries


class Autoencoder(object):
    def __init__(self, database_id, hidden1_units, regularisation, learning_rate0, learning_decay, batch_size_evaluate,
                 batch_size_train, nb_epoch):

        self.database = database_id
        self.hidden1_units = hidden1_units
        self.regularisation = regularisation
        self.learning_rate0 = learning_rate0
        self.learning_decay = learning_decay
        self.batch_size_evaluate = batch_size_evaluate
        self.batch_size_train = batch_size_train
        self.nb_users, self.nb_movies = global_parameters(database=database_id)[0:2]

        self.rmse = 0

        self.epoch_steps = int(self.nb_users / self.batch_size_train)
        self.nb_steps = nb_epoch * self.epoch_steps

        self.Import = Import(database=self.database,
                             test_ratio=0.,
                             validation_ratio=0.1)

        self.Train_set, self.Validation_set, self.Test_set = self.Import.run()

        self.Loss = Loss()

        self.Train = Train(database=self.database,
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
            learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

            prediction = self.Train.inference(x_sparse=x_sparse,
                                         target=target,
                                         hidden1_units=self.hidden1_units)

            loss = self.Loss.full_l2_loss(target=target,
                                          prediction=prediction,
                                          regularisation=self.regularisation,
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
                feed_dict = self.Train.fill_feed_dict_train(target=target,
                                                            x_sparse=x_sparse,
                                                            learning_rate=learning_rate)
                sess.run(train_op,
                         feed_dict=feed_dict)

                summary_str = sess.run(summary_op,
                                       feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                if step % (14 * self.epoch_steps) == 0:
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

