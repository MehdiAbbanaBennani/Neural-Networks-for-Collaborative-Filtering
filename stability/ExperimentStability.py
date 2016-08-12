from autoencoder.Experiment import Experiment

from stability.ImportStability import ImportStability

import time


class ExperimentStability(Experiment):

    def __init__(self, parameters_range, Autoencoder, AutoencoderStability):

        super().__init__(parameters_range=parameters_range, Autoencoder=Autoencoder)

        self.parameters_range['stability']['differences'] = [0]
        self.parameters_range['stability']['rmse'] = [0]

        self.parameters_range['rmse'] = {'autoencoder': 3,
                                         'stability_autoencoder': 3,
                                         'stability_factorisation': 3
                                         }
        self.best_parameters = {'autoencoder': self.select_parameters(self.parameters_range).copy(),
                                'stability_autoencoder': self.select_parameters(self.parameters_range).copy(),
                                'stability_factorisation': self.select_parameters(self.parameters_range).copy()
                                }
        self.AutoencoderStability = AutoencoderStability
        self.log_data = self.log_data_header()
        self.Import = ImportStability(sets_parameters=self.to_scalar(parameters_range['sets']))

    def autoencoder_fixed_parameters(self, parameters):
        rmse_mean = {'autoencoder': 0, 'stability_autoencoder': 0, 'stability_factorisation': 0}

        for i in range(self.parameters_range['experiments']['mean_iterations'][0]):
            sets = self.Import.new_sets(is_test=False)
            rmse = self.three_runs(sets, parameters)
            rmse_mean['autoencoder'] += rmse['autoencoder']
            rmse_mean['stability_factorisation'] += rmse['stability_factorisation']
            rmse_mean['stability_autoencoder'] += rmse['stability_autoencoder']

        rmse_mean['autoencoder'] /= self.parameters_range['experiments']['mean_iterations'][0]
        rmse_mean['stability_factorisation'] /= self.parameters_range['experiments']['mean_iterations'][0]
        rmse_mean['stability_autoencoder'] /= self.parameters_range['experiments']['mean_iterations'][0]

        parameters['rmse'] = rmse_mean
        self.record_data(parameters=parameters)
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
        rmse_stability_autoencoder = self.run_autoencoder(parameters=parameters,
                                                          sets=sets,
                                                          Autoencoder=self.AutoencoderStability)
        # TODO fix the difference

        return {'autoencoder': rmse,
                'stability_factorisation': rmse_stability_factorisation,
                'stability_autoencoder': rmse_stability_autoencoder
                }

    def test_set_evaluation_stability(self):
        for key in self.best_parameters:
            self.best_parameters[key]['autoencoder']['is_test'] = True

            sets = self.Import.new_sets(is_test=True)
            rmse = self.three_runs(parameters=self.best_parameters[key],
                                   sets=sets)

            self.best_parameters[key]['rmse'] = rmse
            self.record_data(parameters=self.best_parameters[key])

    def best_parameters_search(self):
        for i in range(self.parameters_range['experiments']['nb_draws'][0]):
            parameters = self.select_parameters(parameters_range=self.parameters_range)
            parameters['autoencoder']['is_test'] = False

            rmse = self.autoencoder_fixed_parameters(parameters=parameters)
            for key in self.best_parameters:
                if rmse[key] < self.best_parameters[key]['rmse'][key]:
                    parameters['rmse'] = rmse
                    self.best_parameters[key] = parameters

    def autoencoder_experiment(self):
        self.best_parameters_search()
        self.test_set_evaluation_stability()
