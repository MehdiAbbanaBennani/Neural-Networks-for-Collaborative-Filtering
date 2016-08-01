from Autoencoder.Import import Import


class ImportFactorisation(Import):

    def __init__(self, validation_ratio, test_ratio, database_id):
        super().__init__(validation_ratio=validation_ratio,
                         test_ratio=test_ratio,
                         database=database_id)

    def run(self):
        full_dataset = self.full_import(database_id=self.database)
        train, validation, test = self.split_dataset(dataset=full_dataset)
        return train, validation, test

