import numpy as np
state = np.array(
        [
            [0.0, 10, -0.5, -0.5, 0.0, 0.0],
            [0.5, 10, -0.5, -0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 10.0],
            [1.0, 0.0, 0.0, 0.5, 2.0, 10.0],
            [2.0, 0.0, 0.0, 0.5, 3.0, 10.0],
            [3.0, 0.0, 0.0, 0.5, 4.0, 10.0],
        ]
    )
tau = 5 * np.ones(state.shape[0])
bruh = np.concatenate((state, np.expand_dims(tau, -1)), axis=-1)
print(bruh)
print(tau)