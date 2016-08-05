from autoencoder.Experiment import Experiment

from stability.ImportStability import ImportStability

import numpy as np


class ExperimentStability(Experiment):

    def __init__(self, parameters_range, Autoencoder, AutoencoderStability):
        parameters_range['stability']['differences'] = [0]
        parameters_range['stability']['rmse'] = [0]
        super().__init__(parameters_range=parameters_range, Autoencoder=Autoencoder)

        self.Import = ImportStability(sets_parameters=self.to_scalar(parameters_range['sets']))
        self.AutoencoderStability = AutoencoderStability

    def log_data_header(self):
        header = []
        for key in self.parameters_range:
            header = np.append(header, np.array(list(self.parameters_range[key].keys())))
        header = np.append(header, 'rmse')
        header = np.append(header, 'rmse_stability_factorisation')
        header = np.append(header, 'rmse_stability_autoencoder')
        return header

    def autoencoder_fixed_parameters(self, parameters):
        rmse_mean = 0
        rmse_mean_stability_factorisation = 0
        rmse_mean_stability_autoencoder = 0

        for i in range(self.parameters_range['experiments']['mean_iterations'][0]):
            sets = self.Import.new_sets(is_test=False)
            rmse, rmse_stability_factorisation, rmse_mean_stability_autoencoder = self.three_runs(sets, parameters)
            rmse_mean += rmse
            rmse_mean_stability_factorisation += rmse_stability_factorisation
            rmse_mean_stability_autoencoder += rmse_mean_stability_autoencoder

        rmse_mean /= self.parameters_range['experiments']['mean_iterations'][0]
        rmse_mean_stability_factorisation /= self.parameters_range['experiments']['mean_iterations'][0]
        rmse_mean_stability_autoencoder /= self.parameters_range['experiments']['mean_iterations'][0]
        self.record_data(parameters=parameters,
                         rmse=[rmse_mean, rmse_mean_stability_factorisation, rmse_mean_stability_autoencoder])
        return rmse_mean

    def three_runs(self, sets, parameters):
        rmse = self.run_autoencoder(parameters=parameters,
                                    sets=sets,
                                    Autoencoder=self.Autoencoder)

        parameters['stability']['first_learning'] = 'factorisation'
        rmse_stability_factorisation = self.run_autoencoder(parameters=parameters,
                                                            sets=sets,
                                                            Autoencoder=self.AutoencoderStability)

        parameters['stability']['first_learning'] = 'autoencoder'
        rmse_mean_stability_autoencoder = self.run_autoencoder(parameters=parameters,
                                                               sets=sets,
                                                               Autoencoder=self.AutoencoderStability)
        # TODO fix the difference

        return rmse, rmse_stability_factorisation, rmse_mean_stability_autoencoder

    def test_set_evaluation(self, parameters):
        parameters['autoencoder']['is_test'] = True
        sets = self.Import.new_sets(is_test=True)
        rmse = self.run_autoencoder(parameters=parameters, sets=sets, Autoencoder=self.Autoencoder)
        rmse_stability = self.run_autoencoder(parameters=parameters, sets=sets, Autoencoder=self.AutoencoderStability)
        self.record_data(parameters=parameters, rmse=[rmse, rmse_stability])
        return rmse

