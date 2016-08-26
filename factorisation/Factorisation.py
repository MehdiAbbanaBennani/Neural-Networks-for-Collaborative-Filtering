from tools.tools import global_parameters
import numpy as np
import math
from scipy.sparse import csr_matrix


class Factorisation(object):
    def __init__(self, factorisation_sets, factorisation_parameters, sets_parameters):

        if sets_parameters['learning_type'] == 'U':
            self.nb_users, self.nb_movies = global_parameters(sets_parameters=sets_parameters)[0:2]
        else:
            self.nb_movies, self.nb_users = global_parameters(sets_parameters=sets_parameters)[0:2]

        self.dimension = factorisation_parameters['dimension']
        self.iterations = factorisation_parameters['iterations']
        self.landa = factorisation_parameters['landa']

        self.train_set = factorisation_sets[0]
        self.TrainSet_movies = factorisation_sets[0].transpose(copy=True).tocsr()
        self.validation_set = factorisation_sets[1]

        self.R = np.empty((self.nb_users, self.nb_movies))
        self.U = np.random.rand(self.dimension, self.nb_users)
        self.V = np.random.rand(self.nb_movies, self.dimension)

        self.rmse, self.difference_matrix = self.run()
        
    def initializing(self):
        self.U *= 0.1
        self.V *= 0.1
        for i in range(0, self.nb_users):
            self.U[0, i] = 1
        for i in range(0, self.nb_movies):
            self.V[i, 0] = 1
    
    def training(self):
        iteration = 0
        I = np.identity(self.dimension)
        
        while iteration < self.iterations:
            # Estimation of U
            for i in range(0, self.nb_users):
                indices = self.train_set.getrow(i).indices

                self.U[:, i] = np.linalg.solve(np.dot(np.transpose(self.V[indices, :]), 
                                                      self.V[indices, :]) + self.landa * I,
                                               np.dot(np.transpose(self.V[indices, :]),
                                                      np.transpose(self.train_set.getrow(i).data)))
            # Estimation of V
            for i in range(0, self.nb_movies):
                indices = self.TrainSet_movies.getrow(i).indices

                S = np.linalg.solve(np.dot(self.U[:, indices], np.transpose(self.U[:, indices])) + self.landa * I,
                                    np.dot(self.U[:, indices], self.TrainSet_movies.getrow(i).data))
                self.V[i, :] = np.transpose(S)
            iteration += 1

    def evaluation(self, DataSet):
        differences = self.differences(DataSet)
        differences = np.power(differences, 2)
        error = np.mean(differences)
        error = math.sqrt(error)
        return error

    def differences(self, DataSet):
        row_indices = self.sparse_indices(DataSet)
        differences = []
        U_transpose = self.U.transpose().copy()

        for s in range(0, np.size(DataSet.data)):
            i = int(row_indices[s])
            j = int(DataSet.indices[s])
            RT = DataSet.data[s]
            Rij = np.dot(U_transpose[i, :], self.V[j, :])
            differences.append((RT - Rij))
        return differences

    def difference_matrix_build(self, Dataset):
        differences_values = self.differences(Dataset)
        indices_rows = self.sparse_indices(Dataset)
        indices_cols = Dataset.indices
        return csr_matrix((differences_values, (indices_rows, indices_cols)), shape=(self.nb_users, self.nb_movies))

    def run(self):
        print("Factorisation")
        self.initializing()

        print("Training ...")
        self.training()
        print("Training complete")

        print("Evaluating")
        rmse = self.evaluation(self.train_set)
        print("Evaluation complete")

        difference_matrix = self.difference_matrix_build(self.train_set)
        return rmse, difference_matrix
    
    @staticmethod
    def sparse_indices(matrix):
        index = 0
        indices = []
        for k in range(matrix.shape[0]):
            length = matrix.indptr[index + 1] - matrix.indptr[index]
            to_add = np.ones(length) * index
            indices = np.append(indices, to_add)
            index += 1
        return indices

# TODO remove np.append