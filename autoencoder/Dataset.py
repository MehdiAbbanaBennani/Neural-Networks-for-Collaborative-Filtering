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
        matrix = self.category_matrix[name]

        # index = 0
        # for line_number in range(start, end):
        #     new_line_number = self.category_permute[name][line_number]
        #     line = matrix.getrow(new_line_number)
        #
        #     line_values = line.data
        #     line_indices_cols = line.indices
        #     line_indices_lines = np.ones(np.size(line_indices_cols), dtype=int) * index
        #
        #     indices_1.extend(line_indices_lines)
        #     indices_2.extend(line_indices_cols)
        #     ratings.extend(line_values)
        #     index += 1

        indices_1 += [x for list0 in [(np.ones(matrix.getrow(self.category_permute[name][i]).nnz, dtype=int) * (i - start))
                                      for i in range(start, end)]
                      for x in list0]
        indices_2 += [x for list0 in [matrix.getrow(self.category_permute[name][i]).indices for i in range(start, end)]
                      for x in list0]
        ratings += [x for list0 in [matrix.getrow(self.category_permute[name][i]).data for i in range(start, end)]
                    for x in list0]

        ratings = np.asarray(ratings, dtype=np.float32)
        indices = np.asarray(list(zip(indices_1, indices_2)), dtype=int)

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
