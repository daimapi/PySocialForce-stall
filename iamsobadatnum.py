import numpy as np
#state = np.array(
#        [
#            [0.0, 10, -0.5, -0.5, 0.0, 0.0],
#            [0.5, 10, -0.5, -0.5, 0.5, 0.0],
#            [0.0, 0.0, 0.0, 0.5, 1.0, 10.0],
#            [1.0, 0.0, 0.0, 0.5, 2.0, 10.0],
#            [2.0, 0.0, 0.0, 0.5, 3.0, 10.0],
#            [3.0, 0.0, 0.0, 0.5, 4.0, 10.0],
#        ]
#    )
#tau = 5 * np.ones(state.shape[0])
#bruh = np.concatenate((state, np.expand_dims(tau, -1)), axis=-1)
#print(bruh)
#print(tau)
#print(np.expand_dims(tau, -1))
#print(tau.T)
#->[[ 0.  10.  -0.5 -0.5  0.   0.   5. ]
#   [ 0.5 10.  -0.5 -0.5  0.5  0.   5. ]
#   [ 0.   0.   0.   0.5  1.  10.   5. ]
#   [ 1.   0.   0.   0.5  2.  10.   5. ]
#   [ 2.   0.   0.   0.5  3.  10.   5. ]
#   [ 3.   0.   0.   0.5  4.  10.   5. ]]
#  [5. 5. 5. 5. 5. 5.]

initial_state = np.array(
        [
            [0.0, 10, -0.5, -0.5],
            [0.5, 10, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.5],
            [2.0, 0.0, 0.0, 0.5],
            [3.0, 0.0, 0.0, 0.5],
        ]
    )
initial_goals = np.array(
    [
        [[0.0, 0.0, 3], [0.0, 10, 2]],
        [[0.5, 0.0, 3], [0.5, 10, 3]],
        [[1.0, 10.0, 3], [0.0, 0.0, 1]],
        [[2.0, 10.0, 3], [1.0, 0.0, 1]],
        [[3.0, 10.0, 3], [2.0, 0.0, 1]],
        [[4.0, 10.0, 3], [3.0, 0.0, 1]],
    ]
)
sus = np.concatenate(
                (
                initial_state, np.reshape(
                    initial_goals[:, 0], (initial_state.shape[0], 3)
                    )
                )
            , axis = -1)
#print(initial_goals[:, 1, 0:2]) #[:, n, pos(x,y)]
#[[ 0.  10. ]
# [ 0.5 10. ]
# [ 0.   0. ]
# [ 1.   0. ]
# [ 2.   0. ]
# [ 3.   0. ]]
#print(initial_goals[:, 1, 2]) #[:, n, t]
#[2. 3. 1. 1. 1. 1.]
print(sus)
sus2 = np.concatenate((sus, np.expand_dims(np.ones(sus.shape[0]), -1)), axis=-1)
print(sus2)
arrfmask = np.array([True, True, False, True, False, False])
print(sus2[:, 7:8][arrfmask])
for n in sus2[:, 7:8][arrfmask]:
    n = int(n.tolist()[0])
    print(type(n))
print(initial_goals.shape[1])