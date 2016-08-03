import time
import tensorflow as tf
import numpy as np

from autoencoder.Experiment import Experiment

from stability.AutoencoderStability import AutoencoderStability
from stability.ImportStability import ImportStability


class ExperimentStability(Experiment):

    def __init__(self, experiment_parameters, parameters_ranges, sets_parameters):
        super().__init__(experiment_parameters=experiment_parameters,
                         autoencoder_parameters_range=parameters_ranges['autoencoder'],
                         sets_parameters=sets_parameters)
        self.factorisation_parameters_range = parameters_ranges['factorisation']
        self.stability_parameters_range = parameters_ranges['stability']
        self.log_data = (np.append(np.array(list(parameters_ranges['stability'].keys())), 'rmse'))

    def run_autoencoder_stability(self, parameters, sets):
        start_time = time.time()
        tf.reset_default_graph()
        Autoencoder0 = AutoencoderStability(autoencoder_parameters=parameters['autoencoder'],
                                            autoencoder_sets=sets['autoencoder'],
                                            factorisation_sets=sets['factorisation'],
                                            sets_parameters=self.sets_parameters,
                                            factorisation_parameters=parameters['factorisation'],
                                            stability_parameters=parameters['stability'])
        Autoencoder0.run_training()
        rmse = Autoencoder0.rmse
        print(rmse)
        print("--- %s seconds ---" % (time.time() - start_time))
        del Autoencoder0
        return rmse

    def autoencoder_fixed_parameters_stablity(self, parameters):
        rmse_mean = 0
        sets = {}
        for i in range(self.experiment_parameters['mean_iterations']):
            sets['factorisation'], sets['autoencoder'] = ImportStability(sets_parameters=self.sets_parameters).run()
            rmse = self.run_autoencoder_stability(parameters=parameters,
                                                  sets=sets)
            rmse_mean += rmse
        rmse_mean /= self.experiment_parameters['mean_iterations']
        self.record_data_stability(parameters=parameters['stability'],
                                   rmse=rmse_mean)
        return rmse_mean

    def best_parameters_search_stablity(self):
        parameters = {}
        for i in range(self.experiment_parameters['nb_draws']):
            parameters['autoencoder'] = self.select_parameters(parameters_range=self.autoencoder_parameters_range)
            parameters['stability'] = self.select_parameters(parameters_range=self.stability_parameters_range)
            parameters['factorisation'] = self.select_parameters(parameters_range=self.factorisation_parameters_range)
            rmse = self.autoencoder_fixed_parameters_stablity(parameters=parameters)
            if rmse < self.best_rmse:
                best_parameters = parameters
        return best_parameters

    def test_set_evaluation(self):
        pass
        # TODO Test experiments

    def autoencoder_experiment_stability(self):
        best_parameters = self.best_parameters_search_stablity()
        print(best_parameters)

    def run(self):
        self.autoencoder_experiment_stability()
        self.log_to_file()

    def record_data_stability(self, parameters, rmse):
        parameters_array = np.array(list(parameters.values()))
        parameters_array = np.append(parameters_array, rmse)
        parameters_array = parameters_array.astype('|S5')
        self.log_data = np.vstack((self.log_data, parameters_array))




