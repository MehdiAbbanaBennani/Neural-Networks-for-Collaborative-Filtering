from stability.ExperimentStability import ExperimentStability

autoencoder_parameters_range = {'hidden1_units': [700],
                                'regularisation': [0.02],
                                'learning_rate0': [0.001],
                                'learning_decay': [0.9],
                                'batch_size_evaluate': [100],
                                'batch_size_train': [35],
                                'nb_epoch': [2]
                                }

sets_parameters = {'database_id': 0,
                   'test_ratio': 0.1,
                   'validation_ratio': 0.1
                   }

experiment_parameters = {'mean_iterations': 1,
                         'nb_draws': 1
                         }

factorisation_parameters = {'landa': [3],
                            'iterations': [10],
                            'dimension': [10]
                            }

stability_parameters = {'probability': [0.5, 0.6, 0.7, 0.8, 0.9],
                        'subsets_number': [3],
                        'landa_array': [[0.5, 0.3, 0.15, 0.05]]
                        }

parameters_ranges = {'autoencoder': autoencoder_parameters_range,
                     'factorisation': factorisation_parameters,
                     'stability': stability_parameters
                     }

Experiment = ExperimentStability(experiment_parameters=experiment_parameters,
                                 parameters_ranges=parameters_ranges,
                                 sets_parameters=sets_parameters)
Experiment.run()


