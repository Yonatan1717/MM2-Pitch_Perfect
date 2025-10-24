import numpy as np

# Create two 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenate along axis 0 (default for 1D arrays)
result_1d = np.concatenate((arr1, arr2))
print("Concatenated 1D array:", result_1d)

# Create two 2D arrays
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Concatenate along axis 0 (row-wise)
result_axis0 = np.concatenate((matrix1, matrix2), axis=0)
print("\nConcatenated along axis 0:\n", result_axis0)

# Concatenate along axis 1 (column-wise)
result_axis1 = np.concatenate((matrix1, matrix2), axis=1)
print("\nConcatenated along axis 1:\n", result_axis1)