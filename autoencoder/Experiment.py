import time
import numpy as np
import tensorflow as tf

from autoencoder.Autoencoder import Autoencoder
from autoencoder.Import import Import
from tools.tools import log_folder


class Experiment(object):
    def __init__(self, parameters, parameters_range):
        self.parameters = parameters
        self.parameters_range = parameters_range

        self.best_rmse = 3
        self.best_parameters = {}
        self.log_data = self.log_data_header()

    def log_data_header(self):
        header = []
        for key in self.parameters_range:
            header = np.append(header, np.array(list(self.parameters_range[key].keys())))
        header = np.append(header, 'rmse')
        return header

    def record_data(self, parameters, rmse):
        for key in parameters:
            parameters_array = np.array(list(parameters[key].values()))
            parameters_array = np.append(parameters_array, rmse)
            parameters_array = parameters_array.astype('|S5')
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
                else:
                    parameters[key][key_2] = self.pick_line(range)
        return parameters

    @staticmethod
    def pick_line(array):
        line = np.random.randint(0, np.shape(array)[0])
        return array[line, :]

    def run_autoencoder(self, parameters, sets):
        start_time = time.time()
        tf.reset_default_graph()
        Autoencoder1 = Autoencoder(autoencoder_parameters=parameters['autoencoder'],
                                   autoencoder_sets=sets['autoencoder'],
                                   sets_parameters=self.parameters['sets'])
        Autoencoder1.run_training()
        rmse = Autoencoder1.rmse
        print(rmse)
        print("--- %s seconds ---" % (time.time() - start_time))
        del Autoencoder1
        return rmse

    def autoencoder_fixed_parameters(self, parameters):
        rmse_mean = 0
        sets = {}
        for i in range(self.parameters['experiments']['mean_iterations']):
            sets['autoencoder'] = Import(sets_parameters=self.parameters['sets']).run()
            rmse = self.run_autoencoder(parameters=parameters,
                                        sets=sets)
            rmse_mean += rmse
        rmse_mean /= self.parameters['experiments']['mean_iterations']
        self.record_data(parameters=parameters,
                         rmse=rmse_mean)
        return rmse_mean

    def best_parameters_search(self):
        for i in range(self.parameters['experiments']['nb_draws']):
            parameters = self.select_parameters(parameters_range=self.parameters_range)
            rmse = self.autoencoder_fixed_parameters(parameters=parameters)
            if rmse < self.best_rmse:
                best_parameters = parameters
        return best_parameters

    def test_set_evaluation(self):
        pass
        # TODO Test experiments

    def autoencoder_experiment(self):
        best_parameters = self.best_parameters_search()
        print(best_parameters)

    def run(self):
        self.autoencoder_experiment()
        self.log_to_file()
