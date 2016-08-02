#!/usr/bin/python3

import time

import numpy as np
import tensorflow as tf

from autoencoder.Autoencoder import Autoencoder
from tools import log_folder


def autoencoder_experiment(nb_tests, mean_iterations, database, nb_epoch):

    hidden1_array = list(range(500, 700, 20))
    learning_rate_array = np.array([0.00001, 0.0005, 0.0001, 0.00005, 0.001])
    learning_decay_array = np.array([0.01, 0.1, 0.5, 0.7, 0.05])
    weight_decay_array = np.array([0.1, 0.01, 0.5, 0.05])
    iteration = 0

    log_dir = log_folder()
    f = open(log_dir + 'log.txt', 'w')
    f.write("learning_decay \t hidden1 \tlanda \tlearning_rate \tRMSE \n")
    print("learning_decay \t hidden1 \tlanda \tlearning_rate")
    rmse_min = 3

    for s in range(mean_iterations):
        "Training and cross validation tests"
        for k in range(nb_tests):
            iteration += 1
            print(iteration)

            # Parameters generation
            hidden1 = int(np.random.choice(hidden1_array))
            learning_rate = float(np.random.choice(learning_rate_array))
            learning_decay = float(np.random.choice(learning_decay_array))
            weight_decay = float(np.random.choice(weight_decay_array))
            print('\n' + str(learning_decay) + '\t' + str(hidden1) + '\t' + str(weight_decay) + '\t' + str(learning_rate))

            # Running simulation
            start_time = time.time()
            tf.reset_default_graph()
            Autoencoder1 = Autoencoder(database_id=database,
                                       hidden1_units=hidden1,
                                       regularisation=weight_decay,
                                       learning_rate0=learning_rate,
                                       learning_decay=learning_decay,
                                       batch_size_evaluate=100,
                                       batch_size_train=35,
                                       nb_epoch=nb_epoch)
            Autoencoder1.run_training()
            rmse = Autoencoder1.rmse
            print(rmse)
            del Autoencoder1
            print("--- %s seconds ---" % (time.time() - start_time))

            # Storing best parameters
            if rmse < rmse_min:
                best_hidden1 = hidden1
                best_learning_rate = learning_rate
                best_weight_decay = weight_decay
                best_learning_decay = learning_decay
                rmse_min = rmse

            # Writing the results
            f.write(str(learning_decay) + '\t' + str(hidden1) + '\t' + str(weight_decay) + '\t' + str(learning_rate) + '\t' +
                    str(rmse) + '\n')

    # "Validation Test"
    # # Running simulation
    # start_time = time.time()
    # print("The best RMSE")
    # tf.reset_default_graph()
    # Autoencoder1 = Autoencoder(nb_epoch=nb_epoch,
    #                            learning_rate=best_learning_rate,
    #                            hidden1=best_hidden1,
    #                            regularisation=best_weight_decay,
    #                            learning_decay=best_learning_decay,
    #                            batch_size_test=nb_users,
    #                            batch_size_train=35,
    #                            train_sets=train_sets,
    #                            cross_val_sets=cross_val_sets,
    #                            test_sets=test_sets,
    #                            nb_movies=nb_movies,
    #                            nb_users=nb_users,
    #                            train_size=sizes[0],
    #                            cross_val_size=sizes[1],
    #                            test_size=sizes[2],
    #                            is_cross_val=False)
    #
    # Autoencoder1.run()
    # rmse = Autoencoder1.rmse
    # del Autoencoder1
    # print('Best')
    # print(rmse)
    # f.write(str(best_learning_decay) + str(best_hidden1) + '\t' + str(best_weight_decay) + '\t' + str(best_learning_rate) + '\t' +
    #             str(rmse) + '\n')
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # f.close()
