from autoencoder.Import import Import


class ImportStability(Import):
    def __init__(self, sets_parameters):
        super().__init__(validation_ratio=sets_parameters['validation_ratio'],
                         test_ratio=sets_parameters['test_ratio'],
                         database=sets_parameters['database_id'])

    @staticmethod
    def import_factorisation(train, validation):
        train_factorisation = train.copy()
        validation_factorisation = validation.copy()
        return train_factorisation, validation_factorisation

    def run(self):
        full_dataset = self.full_import(database_id=self.database)
        train, validation, test = self.split_dataset(dataset=full_dataset)

        train_factorisation, validation_factorisation = self.import_factorisation(train=train, validation=validation)
        train_normalised_sets, validation_normalised_sets, test_normalised_sets = self.normalise(train, validation, test)

        return [train_factorisation, validation_factorisation],\
               [train_normalised_sets, validation_normalised_sets, test_normalised_sets]