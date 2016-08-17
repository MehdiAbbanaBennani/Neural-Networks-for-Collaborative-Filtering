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

    def normalize_test(self, test_matrix, mean_matrix):

        output_values_ratings = []
        output_values_mean = []

        test_matrix.data -= 3
        test_matrix.data /= 2

        for line_number in range(test_matrix.shape[0]):
            # Normalise the line
            line = test_matrix.getrow(line_number)
            line_train = mean_matrix.getrow(line_number)
            if np.size(line_train) == 0 and np.size(line) > 0:
                line_mean = np.mean(line.data)
                line.data -= line_mean

                # Append to the arrays
                mean_array = np.ones(np.size(line.data)) * line_mean
                output_values_mean.extend(mean_array)
                output_values_ratings.extend(line.data)

            else:
                if np.size(line) > 0:
                    line_train_mean = mean_matrix.getrow(line_number).data[0]
                    line.data -= line_train_mean
                    # assert line.mean() < 2e-1

                    # Append to the arrays
                    mean_array = np.ones(np.size(line.data)) * line_train_mean
                    output_values_mean.extend(mean_array)
                    output_values_ratings.extend(line.data)

        output_values_mean = np.asarray(output_values_mean)
        output_values_ratings = np.asarray(output_values_ratings)

        ratings_sparse = self.to_sparse2(indices=test_matrix.indices,
                                         indptr=test_matrix.indptr,
                                         values=output_values_ratings,
                                         shape=self.shape())
        mean_sparse = self.to_sparse2(indices=test_matrix.indices,
                                      indptr=test_matrix.indptr,
                                      values=output_values_mean,
                                      shape=self.shape())
        return [ratings_sparse, mean_sparse]