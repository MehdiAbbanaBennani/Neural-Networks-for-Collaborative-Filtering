from autoencoder.Experiment import Experiment

from extra_data.ImportExtraData import ImportExtraData

import tensorflow as tf


class ExperimentExtraData(Experiment):
    def __init__(self, parameters_range, Autoencoder):

        super().__init__(parameters_range=parameters_range, Autoencoder=Autoencoder)
        self.Import = ImportExtraData(sets_parameters=self.to_scalar(parameters_range['sets']))
        self.parameters_range['rmse'] = {'autoencoder': 3,
                                         'extra': 3}
        self.log_data = self.log_data_header()

    def run(self):
        for i in range(1, self.parameters_range['experiments']['division'][0], 1):
            self.parameters_range['sets']['train_extra_ratio'][0] = i/10
            self.autoencoder_experiment()
            self.reset_best_parameters()
        self.log_to_file()

    @staticmethod
    def run_autoencoder(parameters, sets, Autoencoder):
        tf.reset_default_graph()
        Autoencoder1 = Autoencoder(parameters=parameters, sets=sets)
        Autoencoder1.run_training()
        rmse = Autoencoder1.rmse
        rmse_extra = Autoencoder1.rmse_extra
        del Autoencoder1
        return rmse, rmse_extra

    def autoencoder_fixed_parameters(self, parameters):
        rmse_mean = 0
        rmse_extra_mean = 0
        self.Import.sets_parameters = parameters['sets']

        for i in range(self.parameters_range['experiments']['mean_iterations'][0]):
            sets = self.Import.new_sets(is_test=False)
            rmse, rmse_extra = self.run_autoencoder(parameters=parameters,
                                                    sets=sets,
                                                    Autoencoder=self.Autoencoder)
            rmse_mean += rmse
            rmse_extra_mean += rmse_extra

        rmse_mean /= self.parameters_range['experiments']['mean_iterations'][0]
        rmse_extra_mean /= self.parameters_range['experiments']['mean_iterations'][0]

        parameters['rmse']['autoencoder'] = rmse_mean
        parameters['rmse']['extra'] = rmse_extra_mean

        print(parameters)

        self.record_data(parameters=parameters)
        return rmse_mean, rmse_extra_mean

    def best_parameters_search(self):
        for i in range(self.parameters_range['experiments']['nb_draws'][0]):
            parameters = self.select_parameters(parameters_range=self.parameters_range)
            parameters['autoencoder']['is_test'] = False
            print(parameters)
            rmse, rmse_extra = self.autoencoder_fixed_parameters(parameters=parameters)

            if rmse < self.best_parameters['autoencoder']['rmse']['autoencoder']:
                parameters['rmse']['autoencoder'] = rmse
                parameters['rmse']['extra'] = rmse_extra
                self.best_parameters['autoencoder'] = parameters

        return self.best_parameters

    def test_set_evaluation(self, parameters):
        parameters_test = parameters['autoencoder']
        parameters_test['autoencoder']['is_test'] = True
        sets = self.Import.new_sets(is_test=True)
        rmse, rmse_extra = self.run_autoencoder(parameters=parameters_test, sets=sets, Autoencoder=self.Autoencoder)
        parameters_test['rmse']['autoencoder'] = rmse
        parameters_test['rmse']['extra'] = rmse_extra
        self.record_data(parameters=parameters_test)
        return rmse, rmse_extra

    def autoencoder_experiment(self):
        best_parameters = self.best_parameters_search()
        print('\n Best parameters')
        print(best_parameters)
        self.test_set_evaluation(best_parameters)

    def reset_best_parameters(self):
        for key, value in self.best_parameters['autoencoder']['rmse'].items():
            self.best_parameters['autoencoder']['rmse'][key] = 3

