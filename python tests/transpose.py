from scipy.sparse import csr_matrix
import numpy as np

row = np.array([0,0,1,2,2,2])
col = np.array([0,2,2,0,1,2])
data = np.array([1,2,3,4,5,6])
data = np.array([1,22,83,4,5,-6])
data = np.array([1,22,83,4,5,-6])

matrix = csr_matrix((data, (row, col)), shape=(3,3) )
matrix2 = csr_matrix((data, (row, col)), shape=(3,3) )
print(matrix.todense())
matrix.transpose(copy=False).tocsr()
print(matrix.todense())
