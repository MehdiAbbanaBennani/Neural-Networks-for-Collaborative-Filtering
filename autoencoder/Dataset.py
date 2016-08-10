import numpy as np


class Dataset(object): # Checked
    def __init__(self, dataset):
        self.ratings = dataset[0]

        self.size = dataset[0].shape[0]
        self.nb_elements = self.ratings.nnz

        self.identity = np.arange(self.size)
        self.permute = np.arange(self.size)

        self.index_completed = np.zeros(4, dtype=int)
        self.last_batch_size = np.zeros(4, dtype=int)
        self.epoch_completed = np.ones(4, dtype=int) * -1

        self.category_indices = {'user': 0, 'rmse': 1}
        self.category_matrix = {'user': self.ratings,
                                'rmse': self.ratings}
        self.category_permute = {'user': self.permute,
                                 'rmse': self.identity}

    def subset(self, start, end, name):
        ratings = []
        indices_1 = []
        indices_2 = []

        index = 0
        for line_number in range(start, end):
            new_line_number = self.category_permute[name][line_number]
            line_values = self.category_matrix[name].getrow(new_line_number).data
            line_indices_cols = self.category_matrix[name].getrow(new_line_number).indices
            line_indices_lines = np.ones(np.size(line_indices_cols), dtype=int) * index
            indices_1.extend(line_indices_lines)
            indices_2.extend(line_indices_cols)
            ratings.extend(line_values)
            index += 1

        indices_1 = np.asarray(indices_1)
        indices_2 = np.asarray(indices_2)
        ratings = np.asarray(ratings)

        ratings = ratings.astype(np.float32)
        indices = np.asarray(list(zip(indices_1.astype(int), indices_2.astype(int))))
        return indices, ratings

    def next_range(self, batch_size, name):
        index_category = self.category_indices[name]
        start = self.index_completed[index_category]
        end = self.index_completed[index_category] + batch_size
        self.index_completed[index_category] += batch_size

        if end > self.size:
            start = 0
            end = batch_size
            self.epoch_completed[index_category] += 1
            np.random.shuffle(self.permute)
        return start, end

    def next_batch(self, batch_size, name):
        start, end = self.next_range(batch_size, name)
        indices, values = self.subset(start, end, name)
        return indices, values

    def reset(self, name):
        category = self.category_indices[name]
        self.index_completed[category] = 0

    @staticmethod
    def sparse_indices(matrix):
        index = 0
        indices = []
        for k in range(matrix.shape[0]):
            length = matrix.indptr[index + 1] - matrix.indptr[index]
            to_add = np.ones(length) * index
            indices.extend(to_add)
            index += 1
        indices = np.asarray(indices)
        return indices
