from autoencoder.Experiment import Experiment
from autoencoder.Autoencoder import Autoencoder

import numpy as np

autoencoder_parameters_range = {'hidden1_units': [500, 550, 600, 650, 700],
                                'regularisation': [0.01, 0.05, 0.1, 0.3, 0.5, 0.8],
                                'learning_rate0': np.logspace(-4, -3, num=3),
                                'learning_decay': np.arange(start=0.1, step=0.1, stop=1),
                                'batch_size_evaluate': [100],
                                'batch_size_train': [35],
                                'nb_epoch': [15],
                                'is_test': [0]
                                }

sets_parameters = {'database_id': [1],
                   'test_ratio': [0.1],
                   'validation_ratio': [0.1]
                   }

experiment_parameters = {'mean_iterations': [1],
                         'nb_draws': [50]
                         }

parameters_range = {'autoencoder': autoencoder_parameters_range,
                    'experiments': experiment_parameters,
                    'sets': sets_parameters
                    }

Experiment = Experiment(parameters_range=parameters_range,
                        Autoencoder=Autoencoder)
Experiment.run()
