from autoencoder.Import import Import


class ImportStability(Import):
    def __init__(self, sets_parameters):
        super().__init__(sets_parameters=sets_parameters)

    @staticmethod
    def import_factorisation(train, validation):
        train_factorisation = train.copy()
        validation_factorisation = validation.copy()
        return train_factorisation, validation_factorisation

    def run(self):
        full_dataset = self.full_import(database_id=self.database_id)
        train, validation, test = self.split_dataset(dataset=full_dataset)

        train_factorisation, validation_factorisation = self.import_factorisation(train=train, validation=validation)
        train_normalised_sets, validation_normalised_sets, test_normalised_sets = self.normalise(train, validation, test)

        return [train_factorisation, validation_factorisation],\
               [train_normalised_sets, validation_normalised_sets, test_normalised_sets]