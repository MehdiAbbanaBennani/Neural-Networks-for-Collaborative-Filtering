from tools import global_parameters
import numpy as np
import math
from scipy.sparse import csr_matrix


class Factorisation(object):
    def __init__(self, TrainSet, ValidationSet, dimension, iterations, landa, database_id):

        self.nb_users, self.nb_movies = global_parameters(database=database_id)[0:2]

        self.dimension = dimension
        self.iterations = iterations
        self.landa = landa

        self.TrainSet = TrainSet
        self.TrainSet_movies = TrainSet.transpose(copy=True).tocsr()
        self.ValidationSet = ValidationSet

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
                indices = self.TrainSet.getrow(i).indices

                self.U[:, i] = np.linalg.solve(np.dot(np.transpose(self.V[indices, :]), 
                                                      self.V[indices, :]) + self.landa * I,
                                               np.dot(np.transpose(self.V[indices, :]),
                                                      np.transpose(self.TrainSet.getrow(i).data)))
            # Estimation of V
            for i in range(0, self.nb_movies):
                indices = self.TrainSet_movies.getrow(i).indices

                S = np.linalg.solve(np.dot(self.U[:, indices], np.transpose(self.U[:, indices])) + self.landa * I,
                                    np.dot(self.U[:, indices], self.TrainSet_movies.getrow(i).data))
                self.V[i, :] = np.transpose(S)
            iteration += 1
    
    def evaluation(self):
        error = 0
        row_indices = self.sparse_indices(self.ValidationSet)

        for s in range(0, np.size(self.ValidationSet.data)):
            i = int(row_indices[s])
            j = int(self.ValidationSet.indices[s])
            RT = self.ValidationSet.data[s]
            Rij = np.dot(self.U[:, i], self.V[j, :])
            error += math.pow((Rij - RT), 2)
        error /= np.size(self.ValidationSet.data)
        error = math.sqrt(error)
        return error

    def run(self):
        print("Factorisation")
        self.initializing()
        print("Training ...")
        self.training()
        print("Training completed")
        print("Evaluating")
        rmse = self.evaluation()
        print("Evaluation completed")
        print(rmse)
        difference_matrix = self.TrainSet.copy()
        difference_matrix.data -= rmse
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
