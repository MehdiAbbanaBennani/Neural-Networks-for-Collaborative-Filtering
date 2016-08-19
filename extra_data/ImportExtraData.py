from autoencoder.Import import Import

import numpy as np
from scipy.sparse import coo
from sklearn.cross_validation import train_test_split


class ImportExtraData(Import):
    def __init__(self, sets_parameters):
        super().__init__(sets_parameters=sets_parameters)

    @staticmethod
    def merge_sparse(matrix1, matrix2):
        matrix1_coo = matrix1.tocoo()
        matrix2_coo = matrix2.tocoo()

        data = np.concatenate((matrix1_coo.data, matrix2_coo.data))
        rows = np.concatenate((matrix1_coo.row, matrix2_coo.row))
        cols = np.concatenate((matrix1_coo.col, matrix2_coo.col))

        full_coo = coo.coo_matrix((data, (rows, cols)), shape=matrix1.shape)
        return full_coo.tocsr()

    def merge_sets(self, sets_1, sets_2):
        output_sets = []
        for i in range(len(sets_1)):
            output_sets.append(self.merge_sparse(sets_1[i], sets_2[i]))
        return output_sets

    def split_dataset_extra(self, is_test):
        if is_test:
            validation_ratio = 0
        else:
            validation_ratio = self.validation_ratio

        x_train, x_validation, y_train, y_validation = train_test_split(self.train_val['x'],
                                                                        self.train_val['y'],
                                                                        test_size=validation_ratio,
                                                                        random_state=38)

        x_train, x_extra, y_train, y_extra = train_test_split(x_train,
                                                              y_train,
                                                              test_size=(1 - self.sets_parameters['train_extra_ratio']),
                                                              random_state=11)

        return self.to_sparse1(x_train, y_train, self.shape()), \
               self.to_sparse1(x_validation, y_validation, self.shape()), \
               self.to_sparse1(self.test['x'], self.test['y'], self.shape()),\
               self.to_sparse1(x_extra, y_extra, self.shape())

    def new_sets(self, is_test):
        sets = {}
        train, validation, test, extra = self.split_dataset_extra(is_test)

        train_normalised_sets, validation_normalised_sets, test_normalised_sets = self.normalise(train, validation,
                                                                                                 test)
        extra_normalised_sets = self.normalize_test(test_matrix=extra, mean_matrix=train_normalised_sets[1])
        extra_normalised_sets = self.merge_sets(train_normalised_sets, extra_normalised_sets)

        sets['autoencoder'] = [train_normalised_sets, validation_normalised_sets, test_normalised_sets]
        sets['extra'] = [extra_normalised_sets, validation_normalised_sets, test_normalised_sets]

        if self.learning_type == 'U':
            pass
        elif self.learning_type == 'V':
            sets = self.transpose_sets(sets)
        else:
            raise ValueError('The learning type is U or V')

        return sets
