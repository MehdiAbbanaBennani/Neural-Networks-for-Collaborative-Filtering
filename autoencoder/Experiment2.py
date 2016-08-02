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
        return rmse_mean

    def autoencoder_experiment(self, experiment_parameters, sets_parameters, autoencoder_parameters_range):

        for i in range(experiment_parameters['nb_draws']):
            autoencoder_parameters = self.select_parameters(parameters_range=autoencoder_parameters_range)
            rmse = self.autoencoder_fixed_parameters(sets_parameters=sets_parameters,
                                                     autoencoder_parameters=autoencoder_parameters)
            print(rmse)

    def run(self):
        self.autoencoder_experiment(experiment_parameters=self.experiment_parameters,
                                    sets_parameters=self.sets_parameters,
                                    autoencoder_parameters_range=self.autoencoder_parameters_range)




