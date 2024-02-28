#codeing = UTF-8
import numpy as np
import shapely.geometry as shp #dude pylint
import shapely
p1 = shp.Point([1, 1])
coords = [[0, 0], [0, 2], [2, 2], [2, 0]]
shape = shp.Polygon(coords)
print(shape.intersects(p1))

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
############################# 2d*1d & wide偏移test
a = np.array([[1, 2],
              [4, 5]])
b = np.array([1, 2])
print(a*b)
c = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])
c[:,0:2] = c[:,0:2] + a*b
c[:,2:4] = c[:,2:4] - a*b
print(c)
#############################
a = np.array([[1, 2],
              [4, 5]])
b = a + 1#---------------------------------------------------------array + int
c = np.array([[[4, 5], [5, 6], [5, 6], [5, 6]],
              [[0, 0], [1, 1], [0, 1], [1, 0]],
              [[4, 5], [5, 6], [5, 6], [5, 6]]])
a = np.concatenate((a, b), axis=-1)
a = np.concatenate((a, b), axis=-1)
a = np.concatenate((a, b), axis=-1)
#print(c.shape)
pos = np.array([[ 0,  10,  -0.5, -0.5,  0,   0,   3,   1, ],
                [ 0.5, 10, -0.5, -0.5,  0.5,  0, 3,   1, ],
                [ 0,   0,   0,   0.5,  1,  10,   3,   1, ],
                [ 1,   0,   0,   0.5,  2,  10,   3,   1, ],
                [ 1,   0,   0,   0.5,  3,  10,   3,   1, ],
                [ 3,   0,   0,   0.5,  4,  10,   3,   1, ],])
pos = pos[:,0:2]#--------------------------------------------------boolean-indexing check
print(pos)
#######################################################################stateutils.find
ab = a.reshape((a.shape[0], 4, 2))#--------------------------------make shape of array become c (x,y one dim)
ab = c
print(ab)
boollist = []
for sect in ab.tolist():
    booll = []
    for posi in pos.tolist():
        booll.append(shp.Polygon(sect).intersects(shp.Point(posi)))
    boollist.append(booll)
print(np.array(boollist))
########################################################################
####################################################array switching(stateutils.perpend)
##print(area)
#vecs = np.array([[1, 2], [2, 3], [2, 3], [2, 3]])
#print(vecs)
##vecs[:,[0,1]] = vecs[:,[1,0]]
#array1 = vecs[:,0]
#array2 = vecs[:,1]
#array = np.stack((array2, array1), axis=-1)
#print(array)
#vecs[:,[0,1]] = vecs[:,[1,0]]
#print(vecs)
####################################################
############################################1d*2d
#a = np.array([1,2,3,4])
#print(array*np.expand_dims(a, axis=1))
############################################