from autoencoder.Experiment import Experiment
from autoencoder.Autoencoder import Autoencoder

autoencoder_parameters_range = {'hidden1_units': [600],
                                'regularisation': [0.02],
                                'learning_rate0': [0.001],
                                'learning_decay': [0.9],
                                'batch_size_evaluate': [100],
                                'batch_size_train': [35],
                                'nb_epoch': [1],
                                'is_test': [0]
                                }

sets_parameters = {'database_id': [0],
                   'test_ratio': [0.1],
                   'validation_ratio': [0.1]
                   }

experiment_parameters = {'mean_iterations': [1],
                         'nb_draws': [1]
                         }

parameters_range = {'autoencoder': autoencoder_parameters_range,
                    'experiments': experiment_parameters,
                    'sets': sets_parameters
                    }

Experiment = Experiment(parameters_range=parameters_range,
                        Autoencoder=Autoencoder)
Experiment.run()
