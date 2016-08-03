import time
import tensorflow as tf
import numpy as np

from autoencoder.Experiment import Experiment

from stability.AutoencoderStability import AutoencoderStability
from stability.ImportStability import ImportStability


class ExperimentStability(Experiment):

    def __init__(self, parameters, parameters_range):
        super().__init__(parameters, parameters_range)

    def run_autoencoder_stability(self, parameters, sets):
        start_time = time.time()
        tf.reset_default_graph()
        Autoencoder0 = AutoencoderStability(autoencoder_parameters=parameters['autoencoder'],
                                            autoencoder_sets=sets['autoencoder'],
                                            factorisation_sets=sets['factorisation'],
                                            sets_parameters=self.parameters['sets'],
                                            factorisation_parameters=parameters['factorisation'],
                                            stability_parameters=parameters['stability'])
        Autoencoder0.run_training()
        rmse = Autoencoder0.rmse
        print(rmse)
        print("--- %s seconds ---" % (time.time() - start_time))
        del Autoencoder0
        return rmse

    def autoencoder_fixed_parameters_stability(self, parameters):
        rmse_mean = 0
        sets = {}
        for i in range(self.parameters['experiment']['mean_iterations']):
            sets['factorisation'], sets['autoencoder'] = ImportStability(sets_parameters=self.parameters['sets']).run()
            rmse = self.run_autoencoder_stability(parameters=parameters,
                                                  sets=sets)
            rmse_mean += rmse
        rmse_mean /= self.parameters['experiment']['mean_iterations']
        self.record_data_stability(parameters=parameters['stability'],
                                   rmse=rmse_mean)
        return rmse_mean

    def best_parameters_search_stability(self):
        for i in range(self.parameters['experiment']['nb_draws']):
            parameters = self.select_parameters(parameters_range=self.parameters_range)
            rmse = self.autoencoder_fixed_parameters_stability(parameters=parameters)
            if rmse < self.best_rmse:
                best_parameters = parameters
        return best_parameters

    def test_set_evaluation(self):
        pass
        # TODO Test experiments

    def autoencoder_experiment_stability(self):
        best_parameters = self.best_parameters_search_stability()
        print(best_parameters)

    def run(self):
        self.autoencoder_experiment_stability()
        self.log_to_file()

    def record_data_stability(self, parameters, rmse):
        parameters_array = np.array(list(parameters.values()))
        parameters_array = np.append(parameters_array, rmse)
        parameters_array = parameters_array.astype('|S5')
        self.log_data = np.vstack((self.log_data, parameters_array))




