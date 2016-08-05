import time
import numpy as np
import tensorflow as tf

from autoencoder.Import import Import
from tools.tools import log_folder


class Experiment(object):
    def __init__(self, parameters_range, Autoencoder):
        self.parameters_range = parameters_range

        self.best_rmse = 3
        self.best_parameters = {}
        self.log_data = self.log_data_header()
        self.Autoencoder = Autoencoder
        self.Import = Import(sets_parameters=self.to_scalar(parameters_range['sets']))

    def log_data_header(self):
        header = []
        for key, value in sorted(self.parameters_range.items()):
            for key2, value2 in sorted(value.items()):
                header = np.append(header, key2)
        header = np.append(header, 'rmse')
        return header

    # TODO reorder data logging
    def record_data(self, parameters, rmse):
        parameters_array = []
        for key, value in sorted(parameters.items()):
            for key2, value2 in sorted(value.items()):
                parameters_array = np.append(parameters_array, value2)
        parameters_array = np.append(parameters_array, rmse)
        self.log_data = np.vstack((self.log_data, parameters_array))

    def log_to_file(self):
        file_name = log_folder() + 'log.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        with open(file_name, 'w') as f:
            f.write(np.array2string(self.log_data, separator=', '))

    def select_parameters(self, parameters_range):
        parameters = {}
        for key in parameters_range:
            parameters[key] = {}
            for key_2 in parameters_range[key]:
                range = parameters_range[key][key_2]
                if len(np.shape(range)) == 1:
                    parameters[key][key_2] = np.random.choice(parameters_range[key][key_2])
                elif len(np.shape(range)) == 0:
                    parameters[key][key_2] = range
                else:
                    parameters[key][key_2] = self.pick_line(range)
        return parameters

    @staticmethod
    def pick_line(array):
        line = np.random.randint(0, np.shape(array)[0])
        return array[line]

    def run_autoencoder(self, parameters, sets, Autoencoder):
        start_time = time.time()
        tf.reset_default_graph()
        Autoencoder1 = Autoencoder(parameters=parameters, sets=sets)
        Autoencoder1.run_training()
        rmse = Autoencoder1.rmse
        print(rmse)
        print("--- %s seconds ---" % (time.time() - start_time))
        del Autoencoder1
        return rmse

    def autoencoder_fixed_parameters(self, parameters):
        rmse_mean = 0
        for i in range(self.parameters_range['experiments']['mean_iterations'][0]):
            sets = self.Import.new_sets(is_test=False)
            rmse = self.run_autoencoder(parameters=parameters,
                                        sets=sets,
                                        Autoencoder=self.Autoencoder)
            rmse_mean += rmse
        rmse_mean /= self.parameters_range['experiments']['mean_iterations'][0]
        self.record_data(parameters=parameters, rmse=rmse_mean)
        return rmse_mean

    def best_parameters_search(self):
        for i in range(self.parameters_range['experiments']['nb_draws'][0]):
            parameters = self.select_parameters(parameters_range=self.parameters_range)
            parameters['autoencoder']['is_test'] = False
            rmse = self.autoencoder_fixed_parameters(parameters=parameters)
            if rmse < self.best_rmse:
                best_parameters = parameters
        return best_parameters

    def test_set_evaluation(self, parameters):
        parameters['autoencoder']['is_test'] = True
        sets = self.Import.new_sets(is_test=True)
        rmse = self.run_autoencoder(parameters=parameters, sets=sets, Autoencoder=self.Autoencoder)
        self.record_data(parameters=parameters, rmse=rmse)
        return rmse
        # TODO Test experiments

    def autoencoder_experiment(self):
        best_parameters = self.best_parameters_search()
        print('\n Best parameters')
        print(best_parameters)
        rmse_test = self.test_set_evaluation(best_parameters)
        print('rmse_test' + str(rmse_test))

    def run(self):
        self.autoencoder_experiment()
        self.log_to_file()

    @staticmethod
    def to_scalar(dictionary):
        scalar_dictionary = {}
        for key in dictionary:
            scalar_dictionary[key] = dictionary[key][0]
        return scalar_dictionary