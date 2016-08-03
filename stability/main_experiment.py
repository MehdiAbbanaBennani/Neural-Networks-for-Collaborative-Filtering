from stability.ExperimentStability import ExperimentStability

sets_parameters = {'database_id': 0,
                   'test_ratio': 0.1,
                   'validation_ratio': 0.1
                   }

experiment_parameters = {'mean_iterations': 1,
                         'nb_draws': 1
                         }

autoencoder_parameters_range = {'hidden1_units': [700],
                                'regularisation': [0.02],
                                'learning_rate0': [0.001],
                                'learning_decay': [0.9],
                                'batch_size_evaluate': [100],
                                'batch_size_train': [35],
                                'nb_epoch': [2]
                                }

factorisation_parameters_range = {'landa': [3],
                                  'iterations': [10],
                                  'dimension': [10]
                            }

stability_parameters_range = {'probability': [0.5, 0.6, 0.7, 0.8, 0.9],
                              'subsets_number': [3],
                              'landa_array': [[0.5, 0.3, 0.15, 0.05]]
                        }

parameters_ranges = {'autoencoder': autoencoder_parameters_range,
                     'factorisation': factorisation_parameters_range,
                     'stability': stability_parameters_range
                     }

parameters = {'experiment': experiment_parameters,
              'sets': sets_parameters}
parameters_range = {'autoencoder': autoencoder_parameters_range,
                    'factorisation': autoencoder_parameters_range,
                    'stability': autoencoder_parameters_range,
                    }

Experiment = ExperimentStability(parameters=parameters,
                                 parameters_range=parameters_range)
Experiment.run()


