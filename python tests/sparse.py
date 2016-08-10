from scipy.sparse import *
from scipy import *
import numpy as np

row = array([0,0,1,2,2,2])
col = array([0,2,2,0,1,2])
data = array([1,2,3,4,5,6])
matrix = csr_matrix( (data,(row,col)), shape=(3,3) )

array0 = []
# array0.extend(matrix.data)

array0 += [x for list in [matrix.getrow(i).data for i in range(2)] for x in list]
print(array0)
print(1)

# a_list += [x for lst in [fun(item) for item in a_list] for x in lst]

