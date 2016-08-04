from autoencoder.Experiment import Experiment

from stability.ImportStability import ImportStability


class ExperimentStability(Experiment):

    def __init__(self, parameters_range, Autoencoder):
        super().__init__(parameters_range=parameters_range, Autoencoder=Autoencoder)

        self.Import = ImportStability(sets_parameters=self.to_scalar(parameters_range['sets']))





