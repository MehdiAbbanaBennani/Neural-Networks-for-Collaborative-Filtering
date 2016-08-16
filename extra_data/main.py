from extra_data.ImportExtraData import ImportExtraData


autoencoder_parameters_range = {'hidden1_units': [700],
                                'regularisation': [0.2],
                                'learning_rate0': [0.001],
                                'learning_decay': [0.9],
                                'batch_size_evaluate': [100],
                                'batch_size_train': [35],
                                'nb_epoch': [5],
                                'is_test': [0]
                                }

sets_parameters = {'database_id': 0,
                   'test_ratio': 0.1,
                   'validation_ratio': 0.1,
                   'learning_type': 'V',
                   'train_extra_ratio': 0.5
                   }

experiment_parameters = {'mean_iterations': [1],
                         'nb_draws': [2]
                         }

parameters_range = {'autoencoder': autoencoder_parameters_range,
                    'experiments': experiment_parameters,
                    'sets': sets_parameters
                    }

Import = ImportExtraData(sets_parameters=sets_parameters)
new_sets = Import.new_sets(is_test=0)