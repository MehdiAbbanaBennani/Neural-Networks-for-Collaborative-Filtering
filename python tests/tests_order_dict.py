import numpy as np
log_data = []

sets_parameters = {'database_id': [0],
                   'test_ratio': [0.1],
                   'validation_ratio': [0.5]
                   }

experiment_parameters = {'mean_iterations': [1],
                         'nb_draws': [5]
                         }


factorisation_parameters = {'landa': [3],
                            'iterations': [10],
                            'dimension': [15]
                            }

stability_parameters = {'probability': [0.45],
                        'subsets_number': [3],
                        'first_learning': 'factorisation'
                        }

parameters = {'factorisation': factorisation_parameters,
              'stability': stability_parameters,
              'experiments': experiment_parameters,
              'sets': sets_parameters
              }

parameters_array = []
for key, value in sorted(parameters.items()):
        for key2, value2 in sorted(value.items()):
            print(key2)
            print(value2)
    # parameters_array = np.append(parameters_array, np.array(list(parameters[key].values())))
# log_data = np.vstack((log_data, parameters_array))


