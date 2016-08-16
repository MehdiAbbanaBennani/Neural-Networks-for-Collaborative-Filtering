import numpy as np

from scipy.sparse import coo
from scipy.sparse import csr_matrix


def merge_sparse(matrix1, matrix2):
    matrix1_coo = matrix1.tocoo()
    matrix2_coo = matrix2.tocoo()

    data = np.concatenate((matrix1_coo.data, matrix2_coo.data))
    rows = np.concatenate((matrix1_coo.row, matrix2_coo.row))
    cols = np.concatenate((matrix1_coo.col, matrix2_coo.col))

    full_coo = coo.coo_matrix((data, (rows,cols)), shape=matrix1.shape)
    return full_coo.tocsr()

row = np.array([0,0,1,2,2,2])
col = np.array([0,2,2,0,1,2])
data = np.array([1,2,3,4,5,6])

row2 = np.array([0,1,1,2,2,2])
col2 = np.array([1,1,2,0,1,2])
data2 = np.array([1,2,3,4,5,6])

matrix = csr_matrix((data, (row, col)), shape=(3,3) )
matrix2 = csr_matrix((data2, (row2, col2)), shape=(3,3) )

merged = merge_sparse(matrix2, matrix)

print(matrix.todense())
print(matrix2.todense())
print(merged.todense())