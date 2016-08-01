from Autoencoder.Dataset import Dataset
from scipy.sparse import csr_matrix

import numpy as np


class DatasetStability(Dataset):
    def __init__(self, dataset, differences, probability, rmse, subsets_number, landa_array):

        super().__init__(dataset)

        self.differences = differences
        self.probability = probability
        self.rmse = rmse
        self.subsets_number = subsets_number
        self.landa_array = landa_array

    def omega_p_build(self):
        probabilities = np.random.rand(self.nb_elements)
        differences = np.absolute(self.differences)

        set1 = np.where(differences <= self.rmse and probabilities <= self.probability)
        set2 = np.where(differences > self.rmse and probabilities <= (1 - self.probability))
        omega_p_set = np.union1d(set1, set2)
        omega_p_set = np.sort(omega_p_set)
        return omega_p_set

    def random_division(self, set):
        coefficients = np.random.choice(a=self.landa_array[1:(self.nb_elements+1)],
                                        size=np.size(set))
        return coefficients

    def loss_coefficients(self, omega_p_set, omega_p_coefficients):
        loss_coefficients = np.zeros(self.nb_elements)

        # Omega0
        loss_coefficients += self.landa_array[0]

        # Omega prime
        loss_coefficients[omega_p_set] += (1 - self.landa_array[0])

        # Removing omega_k
        loss_coefficients[omega_p_set] = loss_coefficients[omega_p_set] - omega_p_coefficients

        return loss_coefficients

    def coefficients_matrix(self, omega_p_set, loss_coefficients):
        omega_p_indices_cols = self.ratings.indices[omega_p_set]
        omega_p_indices_rows = self.sparse_indices(self.ratings)[omega_p_set]
        return csr_matrix((loss_coefficients, (omega_p_indices_rows, omega_p_indices_cols)),
                          shape=[self.nb_users, self.nb_movies])

    def run(self):
        omega_p_set = self.omega_p_build()
        omega_p_coefficients = self.random_division(omega_p_set)
        loss_coefficients = self.loss_coefficients(omega_p_set=omega_p_set,
                                                   omega_p_coefficients=omega_p_coefficients)
        coefficients_matrix = self.coefficients_matrix(omega_p_set=omega_p_set,
                                                       loss_coefficients=loss_coefficients)
        return coefficients_matrix
