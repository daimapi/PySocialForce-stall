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
#print(initial_goals[:, 1, 0:2]) #[:, n+1, pos(x,y)]
#[[ 0.  10. ]
# [ 0.5 10. ]
# [ 0.   0. ]
# [ 1.   0. ]
# [ 2.   0. ]
# [ 3.   0. ]]
##print(initial_goals[:, 1, 2]) #[:, n+1, t]
##[2. 3. 1. 1. 1. 1.]

#print(initial_goals[:, 1, 2][1])
#print(sus)
sus2 = np.concatenate((sus, np.expand_dims(np.ones(sus.shape[0]), -1)), axis=-1)
#print(sus2)
arrfmask = np.array([True, True, False, True, False, False])
#print(sus2[:, 7:8][arrfmask])
#print(sus2[:, 6:7])
#print(np.concatenate(sus2[:, 6:7], axis=None))

##for n in sus2[:, 7:8][arrfmask]:
##    n = int(n.tolist()[0])
#    sus2[:, 7:8][arrfmask] = np.sum(sus2[:, 7:8][arrfmask], np.expand_dims(np.ones(sus2.shape[0])[arrfmask], axis=1))
#    #print(type(n))
##print(initial_goals.shape[1])



arrivedf_mask = np.array([True, True, False, True, False, False])

num = sus2[:, 7:8]
print(num)
numlenght = initial_goals.shape[1]
print(numlenght)
goal = sus2[:, 4:6]
print(goal)
t1 = sus2[:, 6:7] ###############################
t = np.concatenate(t1, axis=None)
print(t)



print(sus2)
print(num[arrivedf_mask])

a = 0
for n in num[arrivedf_mask]:
    print(a)
    n = int(n.tolist()[0])
    print(n)
    num[np.where(arrivedf_mask)[0][a]] += 1 #dont just put the value in a indexed(>=2) array
    if n <= numlenght :
        n += 1
        print(goal[arrivedf_mask][a], initial_goals[:, n-1, 0:2][arrivedf_mask][a])
        print(t[arrivedf_mask][a], initial_goals[:, n-1, 2:3][arrivedf_mask][a])
        goal[np.where(arrivedf_mask)[0][a]] = initial_goals[:, n-1, 0:2][arrivedf_mask][a]
        t[np.where(arrivedf_mask)[0][a]] = initial_goals[:, n-1, 2][arrivedf_mask][a]
    a += 1
print(num)
print(goal)
print(t)
print(t1)