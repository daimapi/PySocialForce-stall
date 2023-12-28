import numpy as np

num = np.array([[1], [1], [1], [1], [1], [1]])
arrivedf_mask = np.array([True, True, False, True, False, False])
a = 1  # You need to define the value of 'a'

print(np.where(arrivedf_mask)) # (array([0, 1, 3], dtype=int32),)
# Check if 'a' is a valid index

    # Update the value in the original array
num[np.where(arrivedf_mask)[0][a]] += 1

print(num)