import time

import numpy as np
import tensorflow as tf

from autoencoder.Autoencoder import Autoencoder
from autoencoder.Import import Import
from tools import log_folder


class Experiment(object):
    def __init__(self, experiment_parameters, autoencoder_parameters_range, sets_parameters):
        self.experiment_parameters = experiment_parameters
        self.autoencoder_parameters_range = autoencoder_parameters_range
        self.sets_parameters = sets_parameters
        self.best_rmse = 3
        self.best_parameters = {}
        self.log_data = (np.append(np.array(list(autoencoder_parameters_range.keys())), 'rmse'))

    def record_data(self, autoencoder_parameters, rmse):
        parameters_array = np.array(list(autoencoder_parameters.values()))
        parameters_array = np.append(parameters_array, rmse)
        parameters_array = parameters_array.astype('|S5')
        self.log_data = np.vstack((self.log_data, parameters_array))

    def log_to_file(self):
        file_name = log_folder() + 'log.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        with open(file_name, 'w') as f:
            f.write(np.array2string(self.log_data, separator=', '))

    @staticmethod
    def select_parameters(parameters_range):
        parameters = {}
        for key in parameters_range:
            parameters[key] = np.random.choice(parameters_range[key])
        return parameters

    @staticmethod
    def run_autoencoder(autoencoder_parameters, autoencoder_sets, sets_parameters):
        start_time = time.time()
        tf.reset_default_graph()
        Autoencoder1 = Autoencoder(autoencoder_parameters=autoencoder_parameters,
                                   autoencoder_sets=autoencoder_sets,
                                   sets_parameters=sets_parameters)
        Autoencoder1.run_training()
        rmse = Autoencoder1.rmse
        print(rmse)
        print("--- %s seconds ---" % (time.time() - start_time))
        del Autoencoder1
        return rmse

    def autoencoder_fixed_parameters(self, sets_parameters, autoencoder_parameters):
        rmse_mean = 0
        for i in range(self.experiment_parameters['mean_iterations']):
            autoencoder_sets = Import(sets_parameters=sets_parameters).run()
            rmse = self.run_autoencoder(autoencoder_parameters=autoencoder_parameters,
                                        autoencoder_sets=autoencoder_sets,
                                        sets_parameters=sets_parameters)
            rmse_mean += rmse
        rmse_mean /= self.experiment_parameters['mean_iterations']
        self.record_data(autoencoder_parameters=autoencoder_parameters,
                         rmse=rmse_mean)
        return rmse_mean

    def best_parameters_search(self, experiment_parameters, sets_parameters, autoencoder_parameters_range):

        for i in range(experiment_parameters['nb_draws']):
            autoencoder_parameters = self.select_parameters(parameters_range=autoencoder_parameters_range)
            rmse = self.autoencoder_fixed_parameters(sets_parameters=sets_parameters,
                                                     autoencoder_parameters=autoencoder_parameters)
            if rmse < self.best_rmse:
                best_parameters = autoencoder_parameters
        return best_parameters

    def test_set_evaluation(self):
        pass
        # TODO Test experiments

    def autoencoder_experiment(self, experiment_parameters, sets_parameters, autoencoder_parameters_range):
        best_parameters = self.best_parameters_search(experiment_parameters=experiment_parameters,
                                                      sets_parameters=sets_parameters,
                                                      autoencoder_parameters_range=autoencoder_parameters_range)
        print(best_parameters)

    def run(self):
        self.autoencoder_experiment(experiment_parameters=self.experiment_parameters,
                                    sets_parameters=self.sets_parameters,
                                    autoencoder_parameters_range=self.autoencoder_parameters_range)
        self.log_to_file()
