from autoencoder.Experiment import Experiment

autoencoder_parameters_range = {'hidden1_units': [600, 700],
                                'regularisation': [0.02],
                                'learning_rate0': [0.001],
                                'learning_decay': [0.9],
                                'batch_size_evaluate': [100],
                                'batch_size_train': [35],
                                'nb_epoch': [1]
                                }

sets_parameters = {'database_id': 1,
                   'test_ratio': 0.1,
                   'validation_ratio': 0.1
                   }

experiment_parameters = {'mean_iterations': 1,
                         'nb_draws': 1
                         }

parameters = {'experiments': experiment_parameters,
              'sets': sets_parameters}
parameters_range = {'autoencoder': autoencoder_parameters_range}

Experiment = Experiment(parameters=parameters,
                        parameters_range=parameters_range)
Experiment.run()
