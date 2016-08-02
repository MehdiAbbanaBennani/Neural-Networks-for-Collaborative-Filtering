from stability.AutoencoderStability import AutoencoderStability
from stability.ImportStability import ImportStability


autoencoder_parameters = {'hidden1_units': 700,
                          'regularisation': 0.02,
                          'learning_rate0': 0.001,
                          'learning_decay': 0.9,
                          'batch_size_evaluate': 100,
                          'batch_size_train': 35,
                          'nb_epoch': 15}

sets_parameters = {'database_id': 1,
                   'test_ratio': 0.,
                   'validation_ratio': 0.1}

factorisation_parameters = {'landa': 3,
                            'iterations': 10,
                            'dimension': 10}

stability_parameters = {'probability': 0.9,
                        'subsets_number': 3,
                        'landa_array': [0.5, 0.3, 0.15, 0.05]}


Import = ImportStability(sets_parameters=sets_parameters)
factorisation_sets, autoencoder_sets = Import.run()

Autoencoder0 = AutoencoderStability(autoencoder_parameters=autoencoder_parameters,
                                    autoencoder_sets=autoencoder_sets,
                                    factorisation_sets=factorisation_sets,
                                    sets_parameters=sets_parameters,
                                    factorisation_parameters=factorisation_parameters,
                                    stability_parameters=stability_parameters)

Autoencoder0.run_training()
