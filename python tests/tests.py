import numpy as np


def select_parameters(parameters_range):

    parameters = {}
    for key in parameters_range:
        parameters[key] = {}
        for key_2 in parameters_range[key]:
            range = parameters_range[key][key_2]
            if len(np.shape(range)) == 1:
                parameters[key][key_2] = np.random.choice(parameters_range[key][key_2])
            else:
                parameters[key][key_2] = pick_line(range)

    return parameters


def record_data(parameters, rmse):

    for key in parameters:
        parameters_array = np.array(list(parameters[key].values()))
    parameters_array = np.append(parameters_array, rmse)
    return parameters_array

def pick_line(array):
    line = np.random.randint(0, np.shape(array)[0])
    return array[line, :]


def log_data_header(parameters_range):
    header = []
    for key in parameters_range:
        header = np.append(header, np.array(list(parameters_range[key].keys())))
    header = np.append(header, 'rmse')
    return header

autoencoder_parameters_range = {'hidden1_units': [600, 700],
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

experiment_parameters = {'mean_iterations': 2,
                         'nb_draws': 2
                         }

parameters = {'experiments': experiment_parameters,
              'sets': sets_parameters}
parameters_range = {'autoencoder': autoencoder_parameters_range}

param = record_data(parameters=parameters, rmse=3)


header = log_data_header(parameters_range)

x = np.arange(9).reshape((3, 3))
print(pick_line(x))
print(header)

print(1)