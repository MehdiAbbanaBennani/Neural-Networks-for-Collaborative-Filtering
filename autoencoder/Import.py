import numpy as np
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix
from tools import global_parameters


class Import(object):
    def __init__(self, sets_parameters):
        self.validation_ratio = sets_parameters['validation_ratio']
        self.test_ratio = sets_parameters['test_ratio']
        self.nb_users, self.nb_movies = global_parameters(sets_parameters['database_id'])[0:2]
        self.database_id = sets_parameters['database_id']

    @staticmethod
    def full_import(database_id):
        data_file = global_parameters(database_id)[3]
        database = np.genfromtxt(data_file, delimiter=',')[:, 0:3]
        database[:, 0:2] -= 1
        return database

    def split_dataset(self, dataset):
        x_train1, x_test, y_train1, y_test = train_test_split(dataset[:, 0:2],
                                                              dataset[:, 2],
                                                              test_size=self.test_ratio,
                                                              random_state=42)
        x_train, x_validation, y_train, y_validation = train_test_split(x_train1,
                                                                        y_train1,
                                                                        test_size=self.validation_ratio,
                                                                        random_state=38)

        return self.to_sparse1(x_train, y_train, self.shape()), \
               self.to_sparse1(x_validation, y_validation, self.shape()), \
               self.to_sparse1(x_test, y_test, self.shape())

    def normalize_train(self, train_matrix):

        output_values_ratings = []
        output_values_mean = []

        train_matrix.data -= 3
        train_matrix.data /= 2

        for line_number in range(train_matrix.shape[0]):
            # Normalise the line
            line = train_matrix.getrow(line_number)
            line_mean = np.mean(line.data)
            line.data -= line_mean
            assert line.mean() < 1e-10

            # Append to the arrays
            mean_array = np.ones(np.size(line.data)) * line_mean
            output_values_mean = np.append(output_values_mean, mean_array)
            output_values_ratings = np.append(output_values_ratings, line.data)

        ratings_sparse = self.to_sparse2(indices=train_matrix.indices,
                                         indptr=train_matrix.indptr,
                                         values=output_values_ratings,
                                         shape=self.shape())
        mean_sparse = self.to_sparse2(indices=train_matrix.indices,
                                      indptr=train_matrix.indptr,
                                      values=output_values_mean,
                                      shape=self.shape())
        return ratings_sparse, mean_sparse

    def normalize_test(self, test_matrix, mean_matrix):

        output_values_ratings = []
        output_values_mean = []

        test_matrix.data -= 3
        test_matrix.data /= 2

        for line_number in range(test_matrix.shape[0]):
            # Normalise the line
            line = test_matrix.getrow(line_number)
            line_train_mean = mean_matrix.getrow(line_number).data[0]
            line.data -= line_train_mean
            assert line.mean() < 1e-2

            # Append to the arrays
            mean_array = np.ones(np.size(line.data)) * line_train_mean
            output_values_mean = np.append(output_values_mean, mean_array)
            output_values_ratings = np.append(output_values_ratings, line.data)

        ratings_sparse = self.to_sparse2(indices=test_matrix.indices,
                                         indptr=test_matrix.indptr,
                                         values=output_values_ratings,
                                         shape=self.shape())
        mean_sparse = self.to_sparse2(indices=test_matrix.indices,
                                      indptr=test_matrix.indptr,
                                      values=output_values_mean,
                                      shape=self.shape())
        return ratings_sparse, mean_sparse

    def normalise(self, train, validation, test):
        train_normalised_sets = self.normalize_train(train_matrix=train)
        validation_normalised_sets = self.normalize_test(test_matrix=validation, mean_matrix=train_normalised_sets[1])
        test_normalised_sets = self.normalize_test(test_matrix=test, mean_matrix=train_normalised_sets[1])
        return train_normalised_sets, validation_normalised_sets, test_normalised_sets

    def run(self):
        full_dataset = self.full_import(database_id=self.database_id)
        train, validation, test = self.split_dataset(dataset=full_dataset)
        train_normalised_sets, validation_normalised_sets, test_normalised_sets = self.normalise(train, validation, test)
        return [train_normalised_sets, validation_normalised_sets, test_normalised_sets]

    def shape(self):
        nb_users = global_parameters(self.database_id)[0]
        nb_movies = global_parameters(self.database_id)[1]
        shape = (nb_users, nb_movies)
        return shape

    @staticmethod
    def to_sparse1(indices, values, shape):
        return csr_matrix((values, (indices[:, 0], indices[:, 1])), shape=shape)

    @staticmethod
    def to_sparse2(indices, indptr, values, shape):
        return csr_matrix((values, indices, indptr), shape=shape)

    @staticmethod
    def to_sparse3(rows, columns, values, shape):
        return csr_matrix((values, (rows, columns)), shape=shape)