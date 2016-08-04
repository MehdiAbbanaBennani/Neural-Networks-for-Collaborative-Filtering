from stability.ExperimentStability import ExperimentStability
from stability.AutoencoderStability import AutoencoderStability

sets_parameters = {'database_id': [0],
                   'test_ratio': [0.1],
                   'validation_ratio': [0.1]
                   }

experiment_parameters = {'mean_iterations': [1],
                         'nb_draws': [1]
                         }
autoencoder_parameters = {'hidden1_units': [500, 550, 600, 650, 700],
                                'regularisation': [0.02, 0.1],
                                'learning_rate0': [0.001, 0.01, 0.1],
                                'learning_decay': [0.001, 0.005, 0.1, 0.5, 0.9],
                                'batch_size_evaluate': [100],
                                'batch_size_train': [35],
                                'nb_epoch': [15],
                                'is_test': [False]
                                }

factorisation_parameters = {'landa': [3],
                            'iterations': [10],
                            'dimension': [10]
                            }

stability_parameters = {'probability': [0.5, 0.6, 0.7, 0.8, 0.9],
                        'subsets_number': [3],
                        'landa_array': [[0.5, 0.3, 0.15, 0.05]]
                        }

parameters_ranges = {'autoencoder': autoencoder_parameters,
                     'factorisation': factorisation_parameters,
                     'stability': stability_parameters
                     }

parameters_range = {'autoencoder': autoencoder_parameters,
                    'factorisation': autoencoder_parameters,
                    'stability': autoencoder_parameters,
                    'experiments': experiment_parameters,
                    'sets': sets_parameters
                    }

Experiment = ExperimentStability(parameters_range=parameters_range,
                                 Autoencoder=AutoencoderStability)
Experiment.run()


