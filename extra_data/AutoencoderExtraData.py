from autoencoder.Autoencoder import Autoencoder
from autoencoder.Dataset import Dataset
from autoencoder.Evaluation import Evaluation

import time
import tensorflow as tf


class AutoencoderExtraData(Autoencoder):
    def __init__(self, parameters, sets):
        super().__init__(parameters=parameters, sets=sets)
        self.Extra_set = Dataset(sets['extra'][0])
        self.rmse_extra = 0

        self.EvaluationExtra = Evaluation(sets_parameters=parameters['sets'],
                                     batch_size_evaluate=self.batch_size_evaluate,
                                     Train_set=self.Extra_set)

    def run_training(self):

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

        # variable_summaries(loss, 'loss/')
        # variable_summaries(learning_rate, 'learning_rate/')
        # summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
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
            print('Extra Data Eval:')
            self.rmse_extra = self.EvaluationExtra.rmse(sess=sess,
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
            print('Extra Data Eval:')
            self.rmse_extra = self.EvaluationExtra.rmse(sess=sess,
                                                        square_error_batch=square_error,
                                                        x_sparse=x_sparse,
                                                        target=target,
                                                        data_set=self.Test_set,
                                                        is_train=False)
        print(self.rmse_extra)
        print(self.rmse)

        end = time.time()
        print('Evaluation time:' + str(end - start))





        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as f:
        #     f.write(ctf)