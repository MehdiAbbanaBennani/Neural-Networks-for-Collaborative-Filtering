from autoencoder.Import import Import


class ImportStability(Import):
    def __init__(self, sets_parameters):
        super().__init__(sets_parameters=sets_parameters)

    @staticmethod
    def import_factorisation(train, validation, test, is_test):
        train_factorisation = train.copy()
        validation_factorisation = validation.copy()
        test_factorisation = test.copy()
        if is_test:
            return [train_factorisation, test_factorisation]
        else:
            return [train_factorisation, validation_factorisation]

    def new_sets(self, is_test):
        sets = {}
        train, validation, test = self.split_dataset(is_test)
        train_factorisation, validation_factorisation = self.import_factorisation(train=train.copy(),
                                                                                  validation=validation.copy(),
                                                                                  test=test.copy(),
                                                                                  is_test=is_test)
        sets['factorisation'] = [train_factorisation, validation_factorisation]

        if self.learning_type == 'U':
            pass
        elif self.learning_type == 'V':
            train = train.transpose(copy=False).tocsr()
            validation = validation.transpose(copy=False).tocsr()
            test = test.transpose(copy=False).tocsr()
        else:
            raise ValueError('The learning type is U or V')

        train_normalised_sets, validation_normalised_sets, test_normalised_sets = self.normalise(train, validation,
                                                                                                 test)
        sets['autoencoder'] = [train_normalised_sets, validation_normalised_sets, test_normalised_sets]
        return sets

